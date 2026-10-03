"""HTTP transcription service: POST audio, get SraVaani native-script text back, served by a pool of model workers."""
import asyncio
import ipaddress
import multiprocessing
import os
import socket
import secrets
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from kural_stt import settings
from kural_stt import transcribe as tr

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".amr", ".webm"}
CHUNK = 1 << 20
MIN_KEY_LENGTH = 24
MAX_REDIRECTS = 5

state = {"pool": None, "device": None, "workers": 0, "threads": 0, "started": None, "inflight": 0, "served": 0, "failed": 0}
pool_lock = asyncio.Lock()


def new_pool() -> ProcessPoolExecutor:
    """A fresh worker pool, each process holding one model copy."""
    ctx = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(max_workers=state["workers"], mp_context=ctx, initializer=tr._init_worker,
                               initargs=(state["device"], state["threads"]))


async def warm_up(pool: ProcessPoolExecutor) -> None:
    """Start every worker and load its model before the first request arrives."""
    loop = asyncio.get_running_loop()
    await asyncio.gather(*[loop.run_in_executor(pool, tr._ready) for _ in range(state["workers"])])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create and warm the worker pool on startup, shut it down on exit."""
    key = settings.SRAVAANI_API_KEY
    if not key and not settings.SRAVAANI_ALLOW_NO_KEY:
        raise SystemExit("Set SRAVAANI_API_KEY in .env (24+ random characters) before starting the API. "
                         "Only on a fully private network, set SRAVAANI_ALLOW_NO_KEY=1 instead.")
    if key and (len(key) < MIN_KEY_LENGTH or key.startswith("change_me")):
        raise SystemExit(f"SRAVAANI_API_KEY is too weak: use at least {MIN_KEY_LENGTH} random characters, "
                         "e.g. python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
    state["device"] = tr.pick_device()
    state["workers"] = tr.safe_workers(state["device"], max(1, settings.SRAVAANI_WORKERS))
    state["threads"] = tr.threads_per_worker(state["device"], state["workers"])
    state["pool"] = new_pool()
    print(f"loading {state['workers']} SraVaani workers on {state['device']} ({state['threads']} threads each)...", flush=True)
    await warm_up(state["pool"])
    state["started"] = time.time()
    print("ready", flush=True)
    yield
    state["pool"].shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Kural-STT", version="1.0", lifespan=lifespan,
              description="SraVaani-1.0 speech-to-text for Indian-language calls. Returns native-script text; no translation.")


def check_key(authorization: Optional[str], x_api_key: Optional[str]) -> None:
    """Reject the request unless it carries SRAVAANI_API_KEY, when one is configured."""
    expected = settings.SRAVAANI_API_KEY
    if not expected:
        return
    given = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
    if not given or not secrets.compare_digest(given, expected):
        raise HTTPException(status_code=401, detail="Missing or wrong API key. Send it as 'X-API-Key: <key>' or 'Authorization: Bearer <key>'.")


async def save_upload(upload: UploadFile, path: str) -> int:
    """Stream an uploaded file to disk, enforcing the size limit; returns bytes written."""
    limit = settings.SRAVAANI_MAX_UPLOAD_MB * CHUNK
    size = 0
    with open(path, "wb") as out:
        while chunk := await upload.read(CHUNK):
            size += len(chunk)
            if size > limit:
                raise HTTPException(status_code=413, detail=f"File larger than {settings.SRAVAANI_MAX_UPLOAD_MB} MB.")
            out.write(chunk)
    return size


def check_public_url(url: str) -> None:
    """Reject URLs that are not http(s) or that resolve to a private, loopback or otherwise internal address."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise HTTPException(status_code=400, detail="url must be an http:// or https:// address.")
    if settings.SRAVAANI_ALLOW_PRIVATE_URLS:
        return
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror:
        raise HTTPException(status_code=502, detail="Could not resolve the url's host name.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise HTTPException(status_code=400, detail="url points to a private or internal address, which is not allowed.")


def fetch_url(url: str, referer: Optional[str], path: str) -> int:
    """Download a public recording URL to disk, checking every redirect and enforcing the size limit; returns bytes written."""
    limit = settings.SRAVAANI_MAX_UPLOAD_MB * CHUNK
    size = 0
    try:
        for _ in range(MAX_REDIRECTS + 1):
            check_public_url(url)
            r = requests.get(url, headers={"Referer": referer} if referer else {}, stream=True, timeout=120, allow_redirects=False)
            if r.is_redirect and r.headers.get("location"):
                url = urljoin(url, r.headers["location"])
                r.close()
                continue
            break
        else:
            raise HTTPException(status_code=502, detail="Too many redirects.")
        with r:
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Could not download the url: HTTP {r.status_code}")
            with open(path, "wb") as out:
                for chunk in r.iter_content(CHUNK):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(status_code=413, detail=f"Recording larger than {settings.SRAVAANI_MAX_UPLOAD_MB} MB.")
                    out.write(chunk)
    except requests.RequestException as ex:
        raise HTTPException(status_code=502, detail=f"Could not download the url: {type(ex).__name__}")
    if size < 1000:
        raise HTTPException(status_code=502, detail=f"Downloaded file is too small to be audio ({size} bytes).")
    return size


async def run_in_pool(path: str) -> dict:
    """Transcribe one file in the worker pool, rebuilding the pool once if a worker crashed natively."""
    loop = asyncio.get_running_loop()
    for attempt in range(2):
        pool = state["pool"]
        try:
            return await asyncio.wait_for(loop.run_in_executor(pool, tr._work_file, path), timeout=settings.SRAVAANI_REQUEST_TIMEOUT)
        except BrokenProcessPool:
            async with pool_lock:
                if state["pool"] is pool:
                    print("worker crashed natively; restarting the pool", flush=True)
                    state["pool"] = new_pool()
                    await warm_up(state["pool"])
            if attempt:
                raise HTTPException(status_code=500, detail="The model crashed on this audio twice.")
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"Transcription took longer than {settings.SRAVAANI_REQUEST_TIMEOUT} s.")


@app.get("/health")
def health() -> dict:
    """Service status: device, worker count and request counters."""
    return {"status": "ok" if state["started"] else "starting", "model": settings.SRAVAANI_MODEL, "device": state["device"],
            "workers": state["workers"], "threads_per_worker": state["threads"], "busy": state["inflight"],
            "served": state["served"], "failed": state["failed"],
            "uptime_sec": round(time.time() - state["started"]) if state["started"] else 0,
            "auth_required": bool(settings.SRAVAANI_API_KEY)}


@app.post("/transcribe")
async def transcribe(file: Optional[UploadFile] = File(None, description="Audio file: mp3, wav, m4a, ogg, flac..."),
                     url: Optional[str] = Form(None, description="Or a recording URL to download instead of uploading"),
                     referer: Optional[str] = Form(None, description="Referer header for the url download, e.g. https://example.com/"),
                     call_id: Optional[str] = Form(None, description="Optional id echoed back in the response"),
                     authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)) -> dict:
    """Transcribe one recording with SraVaani and return its native-script text."""
    check_key(authorization, x_api_key)
    if bool(file) == bool(url):
        raise HTTPException(status_code=400, detail="Send exactly one of: a 'file' upload or a 'url' form field.")
    if url and not settings.SRAVAANI_ALLOW_URL:
        raise HTTPException(status_code=400, detail="URL downloads are switched off on this server; upload the file instead.")
    if state["inflight"] >= state["workers"] + settings.SRAVAANI_MAX_QUEUE:
        raise HTTPException(status_code=503, detail="Server busy: too many recordings queued. Retry shortly.", headers={"Retry-After": "30"})
    suffix = os.path.splitext(file.filename or "")[1].lower() if file else os.path.splitext(url.split("?")[0])[1].lower()
    suffix = suffix if suffix in AUDIO_EXTENSIONS else ".audio"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="kural_")
    os.close(fd)
    t0 = time.time()
    state["inflight"] += 1
    try:
        size = await save_upload(file, path) if file else await asyncio.to_thread(fetch_url, url, referer, path)
        try:
            result = await run_in_pool(path)
        except HTTPException:
            raise
        except Exception as ex:
            if type(ex).__name__ in ("NoBackendError", "LibsndfileError", "DecodeError"):
                raise HTTPException(status_code=415, detail="Not a readable audio file. Send mp3, wav, m4a, ogg or flac.")
            raise HTTPException(status_code=422, detail=f"Could not transcribe this audio: {type(ex).__name__}: {ex}"[:300])
        state["served"] += 1
        return {"call_id": call_id, **result, "bytes": size, "total_sec": round(time.time() - t0, 2),
                "x_real_time": round(result["audio_sec"] / max(result["infer_sec"], 0.01), 1)}
    except HTTPException:
        state["failed"] += 1
        raise
    finally:
        state["inflight"] -= 1
        try:
            os.remove(path)
        except OSError:
            pass


def main() -> None:
    """Run the service with uvicorn on SRAVAANI_HOST:SRAVAANI_PORT (one web process; the model workers live in its pool)."""
    import uvicorn

    uvicorn.run(app, host=settings.SRAVAANI_HOST, port=settings.SRAVAANI_PORT, workers=1)


if __name__ == "__main__":
    main()
