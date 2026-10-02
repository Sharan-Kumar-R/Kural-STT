"""Paths, model ids, API settings and config loading shared by every step."""
import json
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
GROQ_WORKERS = int(os.getenv("GROQ_WORKERS", "4"))

DATA_DIR = Path(os.getenv("SRAVAANI_DATA_DIR") or BASE_DIR / "data")
AUDIO_DIR = DATA_DIR / "audio"
MANIFEST = DATA_DIR / "manifest.json"
TRANSCRIPTS = DATA_DIR / "sravaani_out.jsonl"
BASELINE = DATA_DIR / "groq_out.jsonl"
COMPARISON = DATA_DIR / "compare_out.jsonl"
ATTEMPTS = DATA_DIR / "attempts.json"
FAULT_LOG = DATA_DIR / "fault.log"

OUTPUT_DIR = Path(os.getenv("SRAVAANI_OUTPUT_DIR") or BASE_DIR / "outputs")
ANALYSIS = OUTPUT_DIR / "analysis.txt"
EXAMPLES = OUTPUT_DIR / "examples.txt"

CONFIG = Path(os.getenv("SRAVAANI_CONFIG") or BASE_DIR / "eval_config.json")

os.environ.setdefault("HF_HOME", str(BASE_DIR / "hf"))
if os.getenv("SRAVAANI_TMP_DIR"):
    os.environ["TMP"] = os.environ["TEMP"] = os.environ["SRAVAANI_TMP_DIR"]


def load_config() -> dict:
    """The evaluation config from eval_config.json, or a clear error when it is missing."""
    if not CONFIG.exists():
        raise SystemExit(f"Copy eval_config.example.json to {CONFIG.name} and fill in your groups.")
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


def group_for(call: dict, config: dict) -> dict:
    """The config group a manifest call belongs to, or an empty dict."""
    for group in config.get("groups", []):
        if group["label"] == call.get("state_group") and group.get("client", call.get("client")) == call.get("client"):
            return group
    return {}


def language_for(call: dict, config: dict) -> str:
    """Spoken language of a call, from the manifest or else from its config group."""
    return call.get("language") or group_for(call, config).get("language") or "an Indian language"


def intents_for(call: dict, config: dict) -> list:
    """Allowed intents for a call: its group's list, else the config-wide list."""
    return group_for(call, config).get("intents") or config.get("intents") or []


def require_groq() -> str:
    """The Groq API key, or a clear error when it is missing."""
    if not GROQ_API_KEY:
        raise SystemExit("Set GROQ_API_KEY in .env.")
    return GROQ_API_KEY
