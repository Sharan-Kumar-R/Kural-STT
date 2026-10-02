"""Step 3: the baseline route, Groq Whisper speech-to-English on every manifest call, resuming where a previous run stopped."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from sravaani_eval import groq_client, settings
from sravaani_eval.store import done_ids, read_json


def baseline_call(call: dict) -> dict:
    """Groq Whisper English for one call, or the error it raised."""
    t0 = time.time()
    try:
        text = groq_client.translate_audio(settings.AUDIO_DIR / call["audio"])
    except Exception as ex:
        return {"call_uuid": call["call_uuid"], "text": None, "error": f"{type(ex).__name__}: {ex}"[:300]}
    return {"call_uuid": call["call_uuid"], "text": text, "api_sec": round(time.time() - t0, 1)}


def main() -> None:
    """Append one JSON line per call to data/groq_out.jsonl."""
    settings.require_groq()
    manifest = read_json(settings.MANIFEST)
    done = done_ids(settings.BASELINE)
    todo = [m for m in manifest if m["call_uuid"] not in done]
    print(f"{len(todo)} of {len(manifest)} to send to Groq {settings.GROQ_STT_MODEL}", flush=True)
    with open(settings.BASELINE, "a", encoding="utf-8") as out, ThreadPoolExecutor(settings.GROQ_WORKERS) as pool:
        for k, row in enumerate(pool.map(baseline_call, todo), 1):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{k}/{len(todo)} {row['call_uuid']} {row.get('api_sec')}s {row.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
