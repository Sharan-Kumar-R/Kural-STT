"""Minimal Groq API client: Whisper speech-to-English and chat completions, with retry on rate limits."""
import json
import time
from pathlib import Path

import requests

from sravaani_eval import settings

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 5
TIMEOUT_SEC = 120


def _post(path: str, **kwargs) -> dict:
    """POST to the Groq API, retrying rate limits and server errors with backoff."""
    headers = {"Authorization": f"Bearer {settings.require_groq()}"}
    url = f"{settings.GROQ_BASE_URL}/{path}"
    for attempt in range(MAX_RETRIES):
        resp = requests.post(url, headers=headers, timeout=TIMEOUT_SEC, **kwargs)
        if resp.status_code not in RETRY_STATUSES:
            resp.raise_for_status()
            return resp.json()
        wait = float(resp.headers.get("retry-after") or 2 ** attempt)
        time.sleep(min(wait, 60))
    resp.raise_for_status()
    return resp.json()


def translate_audio(audio_path: Path) -> str:
    """English text of one recording from Groq Whisper's translation endpoint."""
    with open(audio_path, "rb") as f:
        result = _post(
            "audio/translations",
            files={"file": (audio_path.name, f)},
            data={"model": settings.GROQ_STT_MODEL, "temperature": "0", "response_format": "json"},
        )
    return (result.get("text") or "").strip()


def chat(prompt: str, as_json: bool = False, max_tokens: int = 4000) -> str:
    """Reply text of one single-message chat completion at temperature 0."""
    body = {
        "model": settings.GROQ_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
    }
    if as_json:
        body["response_format"] = {"type": "json_object"}
    result = _post("chat/completions", json=body)
    return (result["choices"][0]["message"].get("content") or "").strip()


def chat_json(prompt: str) -> dict:
    """Parsed JSON object from a JSON-mode chat completion."""
    return json.loads(chat(prompt, as_json=True, max_tokens=1000))
