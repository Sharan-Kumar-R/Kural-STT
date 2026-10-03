"""SraVaani engine: device and worker sizing, model loading, and per-call transcription used by the API worker pool."""
import faulthandler
import os
import time

from kural_stt import settings

SAMPLE_RATE = 16000
PIECE_SEC = 30
MIN_PIECE_SAMPLES = 1600

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
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    faulthandler.enable(file=open(settings.FAULT_LOG, "a"), all_threads=True)
    _device = device
    _model = load_model(device, threads)


def _ready() -> str:
    """Confirm a worker has its model loaded; returns the device it runs on."""
    return _device


def _work_file(path: str) -> dict:
    """Transcribe one audio file at an absolute path inside a worker."""
    return {**transcribe_call(_model, path), "device": _device}
