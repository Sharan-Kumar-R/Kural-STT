"""Settings read from .env: model, workers, API and limits."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
SRAVAANI_MODEL = os.getenv("SRAVAANI_MODEL", "ARTPARK-IISc/SraVaani-1.0")
SRAVAANI_WORKERS = int(os.getenv("SRAVAANI_WORKERS") or 8)
SRAVAANI_DEVICE = (os.getenv("SRAVAANI_DEVICE") or "auto").lower()
SRAVAANI_THREADS = int(os.getenv("SRAVAANI_THREADS") or 0)
SRAVAANI_PIECE_BATCH = int(os.getenv("SRAVAANI_PIECE_BATCH") or 8)
SRAVAANI_WORKER_RAM_GB = float(os.getenv("SRAVAANI_WORKER_RAM_GB") or 4.5)
SRAVAANI_WORKER_GPU_GB = float(os.getenv("SRAVAANI_WORKER_GPU_GB") or 3.5)
SRAVAANI_FP16 = (os.getenv("SRAVAANI_FP16") or "0") == "1"
SRAVAANI_JIT_OPT = (os.getenv("SRAVAANI_JIT_OPT") or ("0" if os.name == "nt" else "1")) == "1"

SRAVAANI_API_KEY = os.getenv("SRAVAANI_API_KEY") or ""
SRAVAANI_HOST = os.getenv("SRAVAANI_HOST") or "127.0.0.1"
SRAVAANI_PORT = int(os.getenv("SRAVAANI_PORT") or 8000)
SRAVAANI_MAX_UPLOAD_MB = int(os.getenv("SRAVAANI_MAX_UPLOAD_MB") or 100)
SRAVAANI_REQUEST_TIMEOUT = int(os.getenv("SRAVAANI_REQUEST_TIMEOUT") or 1800)
SRAVAANI_MAX_QUEUE = int(os.getenv("SRAVAANI_MAX_QUEUE") or 32)
SRAVAANI_ALLOW_URL = (os.getenv("SRAVAANI_ALLOW_URL") or "1") == "1"
SRAVAANI_ALLOW_PRIVATE_URLS = (os.getenv("SRAVAANI_ALLOW_PRIVATE_URLS") or "0") == "1"
SRAVAANI_ALLOW_NO_KEY = (os.getenv("SRAVAANI_ALLOW_NO_KEY") or "0") == "1"

LOG_DIR = Path(os.getenv("SRAVAANI_LOG_DIR") or BASE_DIR / "logs")
FAULT_LOG = LOG_DIR / "fault.log"

os.environ.setdefault("HF_HOME", str(BASE_DIR / "hf"))
if os.getenv("SRAVAANI_TMP_DIR"):
    os.environ["TMP"] = os.environ["TEMP"] = os.environ["SRAVAANI_TMP_DIR"]
