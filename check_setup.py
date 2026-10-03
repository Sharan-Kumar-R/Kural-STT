"""Check that this machine can run SraVaani: Python, PyTorch build, GPU, cores, RAM, token, and a timed test transcription on generated audio."""
import os
import platform
import shutil
import subprocess
import sys
import time

from kural_stt import settings

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
results = []


def report(status: str, item: str, detail: str) -> None:
    """Print one check line and remember it for the summary."""
    results.append(status)
    print(f"[{status}] {item:<22} {detail}", flush=True)


def nvidia_smi() -> str:
    """GPU name, driver, temperature, power and clocks from nvidia-smi, or an empty string."""
    if not shutil.which("nvidia-smi"):
        return ""
    q = "name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks_throttle_reasons.active"
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


def main() -> None:
    """Run every check, then a short timed transcription with the configured worker settings."""
    print(f"SraVaani setup check on {platform.node()} ({platform.system()} {platform.release()})\n")
    py = sys.version_info
    report(OK if py >= (3, 10) else FAIL, "Python", platform.python_version())

    import torch

    cuda_build = torch.version.cuda or "cpu-only"
    report(OK, "PyTorch", f"{torch.__version__} (CUDA build: {cuda_build})")
    cores = os.cpu_count() or 1
    report(OK if cores >= 8 else WARN, "CPU threads", f"{cores} logical cores; {settings.SRAVAANI_WORKERS} workers configured")
    from kural_stt.transcribe import total_ram_gb

    ram = total_ram_gb()
    need = settings.SRAVAANI_WORKERS * settings.SRAVAANI_WORKER_RAM_GB + 4
    report(OK if ram >= need else WARN, "RAM", f"{ram:.0f} GB total; about {need:.0f} GB needed for {settings.SRAVAANI_WORKERS} workers (fewer will be used automatically)")
    report(OK if settings.HF_TOKEN else FAIL, "HF_TOKEN", "set" if settings.HF_TOKEN else "missing: add it to .env and accept the model terms on Hugging Face")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        arch = f"sm_{cap[0]}{cap[1]}"
        archs = torch.cuda.get_arch_list()
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        report(OK, "GPU", f"{name}, {mem:.0f} GB, compute {arch}")
        if arch in archs or any(a.startswith(f"sm_{cap[0]}") for a in archs):
            report(OK, "GPU kernels", f"this PyTorch build includes {arch}")
        else:
            report(FAIL, "GPU kernels", f"this PyTorch build has {archs} but not {arch}. For a V100 install: pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126")
        report(OK if mem >= settings.SRAVAANI_WORKERS * settings.SRAVAANI_WORKER_GPU_GB else WARN, "GPU memory", f"about {settings.SRAVAANI_WORKERS * settings.SRAVAANI_WORKER_GPU_GB:.0f} GB needed for {settings.SRAVAANI_WORKERS} workers")
    else:
        report(WARN, "GPU", f"no CUDA GPU visible to PyTorch (build: {cuda_build}); will run on CPU")
    smi = nvidia_smi()
    if smi:
        report(OK, "nvidia-smi", smi)

    if FAIL in results:
        print("\nFix the FAIL lines above, then run this again.")
        sys.exit(1)

    from kural_stt.transcribe import load_model, pick_device, threads_per_worker

    device = pick_device()
    threads = threads_per_worker(device, settings.SRAVAANI_WORKERS)
    print(f"\nTest transcription on {device}, {threads} threads (one worker), with 120 s of generated audio...")
    model = load_model(device, threads)
    sr = 16000
    t = torch.arange(120 * sr) / sr
    audio = 0.1 * torch.sin(2 * 3.1416 * 220 * t) * (torch.sin(2 * 3.1416 * 0.5 * t) > 0) + 0.01 * torch.randn(t.shape)
    pieces = [audio[i:i + 30 * sr] for i in range(0, len(audio), 30 * sr)]
    with torch.inference_mode():
        model.transcribe(pieces[:1], batch_size=1, return_hypotheses=True)
        t0 = time.time()
        model.transcribe(pieces, batch_size=len(pieces), return_hypotheses=True)
        dt = time.time() - t0
    report(OK, "Single worker speed", f"120 s of audio in {dt:.1f} s = {120 / dt:.0f}x real time")
    if smi:
        report(OK, "nvidia-smi after", nvidia_smi())
    print(f"\nAll checks passed. Expect very roughly up to {settings.SRAVAANI_WORKERS}x that with {settings.SRAVAANI_WORKERS} workers on CPU; "
          "on GPU the gain is smaller. Real calls are the true test: start the API and send a few recordings.")
    print("Generated audio is a tone, not speech, so the transcript text is meaningless; only the speed matters here.")


if __name__ == "__main__":
    main()
