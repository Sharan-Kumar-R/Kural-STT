"""Step 2: transcribe every manifest call with SraVaani in parallel worker processes, on GPU when available, resuming where a previous run stopped."""
import faulthandler
import json
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

from sravaani_eval import settings
from sravaani_eval.store import read_json, read_jsonl, write_json

SAMPLE_RATE = 16000
PIECE_SEC = 30
MIN_PIECE_SAMPLES = 1600
MAX_ATTEMPTS = 2

_model = None
_device = "cpu"


def pick_device() -> str:
    """The configured device, resolving 'auto' to cuda when a GPU is visible."""
    import torch

    if settings.SRAVAANI_DEVICE != "auto":
        return settings.SRAVAANI_DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


def total_ram_gb() -> float:
    """Total system RAM in GB, or 0 when it cannot be read."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (AttributeError, ValueError, OSError):
        try:
            import ctypes

            class Mem(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                            ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("x", ctypes.c_ulonglong)]
            m = Mem()
            m.len = ctypes.sizeof(Mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.total / 1e9
        except Exception:
            return 0.0


def safe_workers(device: str, wanted: int) -> int:
    """Requested worker count, capped so the copies fit in RAM and, on GPU, in GPU memory."""
    import torch

    caps = [wanted]
    ram = total_ram_gb()
    if ram:
        caps.append(int((ram - 4) // settings.SRAVAANI_WORKER_RAM_GB))
    if device.startswith("cuda"):
        caps.append(int(torch.cuda.get_device_properties(0).total_memory / 1e9 // settings.SRAVAANI_WORKER_GPU_GB))
    workers = max(1, min(caps))
    if workers < wanted:
        print(f"capping workers at {workers} (asked {wanted}): {ram:.0f} GB RAM at {settings.SRAVAANI_WORKER_RAM_GB} GB each"
              + (f", GPU at {settings.SRAVAANI_WORKER_GPU_GB} GB each" if device.startswith("cuda") else ""), flush=True)
    return workers


def threads_per_worker(device: str, workers: int) -> int:
    """CPU threads each worker may use: the configured value, else an even split of the cores."""
    if settings.SRAVAANI_THREADS > 0:
        return settings.SRAVAANI_THREADS
    if device.startswith("cuda"):
        return 2
    return max(1, (os.cpu_count() or 1) // workers)


def load_model(device: str = "cpu", threads: int = 0):
    """SraVaani in eval mode on the given device, with TorchScript re-optimisation off when configured (it crashes natively on Windows)."""
    import torch
    from transformers import AutoModel

    if not settings.HF_TOKEN:
        raise SystemExit("Set HF_TOKEN in .env; the SraVaani weights are gated.")
    if threads:
        torch.set_num_threads(threads)
    if not settings.SRAVAANI_JIT_OPT:
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._set_graph_executor_optimize(False)
    kwargs = {"torch_dtype": torch.float16} if settings.SRAVAANI_FP16 and device.startswith("cuda") else {}
    t0 = time.time()
    model = AutoModel.from_pretrained(settings.SRAVAANI_MODEL, trust_remote_code=True, token=settings.HF_TOKEN, **kwargs)
    model = model.to(device).eval()
    print(f"[pid {os.getpid()}] model loaded on {device} with {threads or torch.get_num_threads()} threads in {time.time() - t0:.0f}s", flush=True)
    return model


def transcribe_call(model, audio_path) -> dict:
    """Native-script text of one call plus its audio length, inference time and piece count."""
    import librosa
    import torch

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    step = PIECE_SEC * SAMPLE_RATE
    pieces = [torch.from_numpy(audio[i:i + step].copy()) for i in range(0, len(audio), step) if len(audio[i:i + step]) > MIN_PIECE_SAMPLES]
    t0 = time.time()
    texts = []
    with torch.inference_mode():
        for s in range(0, len(pieces), settings.SRAVAANI_PIECE_BATCH):
            batch = pieces[s:s + settings.SRAVAANI_PIECE_BATCH]
            texts += [h.text for h in model.transcribe(batch, batch_size=len(batch), return_hypotheses=True)]
    return {
        "text": " ".join(t for t in texts if t).strip(),
        "audio_sec": round(len(audio) / SAMPLE_RATE, 1),
        "infer_sec": round(time.time() - t0, 1),
        "pieces": len(pieces),
    }


def _init_worker(device: str, threads: int) -> None:
    """Load one model copy per worker process."""
    global _model, _device
    faulthandler.enable(file=open(settings.FAULT_LOG, "a"), all_threads=True)
    _device = device
    _model = load_model(device, threads)


def _work(uuid: str, audio: str) -> dict:
    """Transcribe one call inside a worker, turning Python errors into an error row."""
    try:
        return {"call_uuid": uuid, **transcribe_call(_model, settings.AUDIO_DIR / audio), "device": _device}
    except Exception as ex:
        return {"call_uuid": uuid, "text": None, "error": f"{type(ex).__name__}: {ex}"[:300]}


def run_batch(calls: list, workers: int, device: str, threads: int, out, attempts: dict, stats: dict) -> None:
    """Keep at most `workers` calls in flight; on a native crash, charge an attempt to every call that was in flight."""
    queue = list(calls)
    inflight = {}
    ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(device, threads))
    try:
        while queue or inflight:
            while queue and len(inflight) < workers:
                m = queue.pop(0)
                inflight[pool.submit(_work, m["call_uuid"], m["audio"])] = m["call_uuid"]
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                row = fut.result()
                uuid = inflight.pop(fut)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                stats["done"] += 1
                stats["audio"] += row.get("audio_sec") or 0
                elapsed = time.time() - stats["t0"]
                print(f"{stats['done']}/{stats['total']} {uuid[:8]} {row.get('audio_sec')}s audio in {row.get('infer_sec')}s | "
                      f"overall {stats['audio'] / max(elapsed, 1e-9):.1f}x real time {row.get('error', '')}", flush=True)
    except BrokenProcessPool:
        for uuid in inflight.values():
            attempts[uuid] = attempts.get(uuid, 0) + 1
        write_json(settings.ATTEMPTS, attempts)
        print(f"worker crashed natively; {len(inflight)} in-flight calls will be retried one at a time", flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    """Append one JSON line per call to data/sravaani_out.jsonl using SRAVAANI_WORKERS parallel workers; calls that failed in an earlier run are retried."""
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_json(settings.MANIFEST)
    device = pick_device()
    workers = safe_workers(device, max(1, settings.SRAVAANI_WORKERS))
    threads = threads_per_worker(device, workers)
    attempts = read_json(settings.ATTEMPTS) if settings.ATTEMPTS.exists() else {}
    stats = {"done": 0, "audio": 0.0, "t0": time.time(), "total": 0}
    started = len(read_jsonl(settings.TRANSCRIPTS))
    with open(settings.TRANSCRIPTS, "a", encoding="utf-8") as out:
        while True:
            rows = read_jsonl(settings.TRANSCRIPTS)
            done = {r["call_uuid"] for r in rows if r.get("text") is not None} | {r["call_uuid"] for r in rows[started:]}
            todo = [m for m in manifest if m["call_uuid"] not in done]
            if not todo:
                break
            for m in [m for m in todo if attempts.get(m["call_uuid"], 0) >= MAX_ATTEMPTS]:
                out.write(json.dumps({"call_uuid": m["call_uuid"], "text": None, "error": "native crash on every attempt"}) + "\n")
                out.flush()
            todo = [m for m in todo if attempts.get(m["call_uuid"], 0) < MAX_ATTEMPTS]
            if not todo:
                continue
            stats["total"] = stats["done"] + len(todo)
            print(f"{len(todo)} of {len(manifest)} to transcribe on {device} with {workers} workers x {threads} threads", flush=True)
            suspects = [m for m in todo if attempts.get(m["call_uuid"], 0) > 0]
            normal = [m for m in todo if attempts.get(m["call_uuid"], 0) == 0]
            if suspects:
                run_batch(suspects, 1, device, threads, out, attempts, stats)
            if normal:
                run_batch(normal, workers, device, threads, out, attempts, stats)
    elapsed = time.time() - stats["t0"]
    if stats["done"]:
        print(f"finished {stats['done']} calls, {stats['audio'] / 3600:.2f} h audio in {elapsed / 60:.1f} min wall time "
              f"= {stats['audio'] / elapsed:.1f}x real time", flush=True)


if __name__ == "__main__":
    main()
