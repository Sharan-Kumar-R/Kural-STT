"""Step 4: translate SraVaani text to English with a Groq LLM, then judge both English transcripts with the same prompt."""
import json
from concurrent.futures import ThreadPoolExecutor

from sravaani_eval import groq_client, settings
from sravaani_eval.store import done_ids, read_json, read_jsonl

TRANSLATE_PROMPT = """Translate this phone-call transcript into English. It is an automatic speech-recognition transcript in {lang}, possibly mixed with English or Hindi words, from {context}. Speakers are not separated.

Translate faithfully, sentence by sentence. Keep numbers, amounts, grams, dates, times, names and places exactly. Do not summarise, add or explain anything. If a stretch is unintelligible, write [unclear].

Output only the English translation.

TRANSCRIPT:
{text}"""

JUDGE_PROMPT = """You review an English transcript of a phone call from {context}. The transcript was produced by automatic speech recognition and may contain errors. Speakers are not separated.

Decide whether it holds a real customer conversation with a request that can be understood.
- NOISE: silence, IVR or hold music only, a wrong number, garbled text with no understandable request, or no customer speaking.
- Otherwise it is real.

{intent_rule}

Reply with a JSON object only:
{{"is_noise": true or false, "noise_reason": "short reason, or null when real", "intent": "the intent, or NOISE", "summary": "one sentence on what the caller wanted, or null"}}

TRANSCRIPT:
{text}"""

MIN_WORDS = 3


def intent_rule(intents: list) -> str:
    """The judge's instruction for naming the intent: a fixed list when configured, else a short free label."""
    if intents:
        return "When real, set intent to exactly one of: " + "; ".join(intents) + "."
    return "When real, set intent to a short label of two to five words for what the caller wanted."


def translate(text: str, lang: str, context: str) -> str:
    """English translation of one SraVaani transcript."""
    return groq_client.chat(TRANSLATE_PROMPT.format(lang=lang, context=context, text=text))


def judge(text: str, context: str, intents: list) -> dict:
    """NOISE verdict, intent and summary for one English transcript."""
    if len((text or "").split()) < MIN_WORDS:
        return {"intent": "NOISE", "noise_reason": "transcript_too_short", "summary": None}
    try:
        result = groq_client.chat_json(JUDGE_PROMPT.format(context=context, intent_rule=intent_rule(intents), text=text))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    noise = bool(result.get("is_noise"))
    intent = "NOISE" if noise else (result.get("intent") or "Unknown")
    if intents and not noise and intent not in intents:
        intent = "Unknown"
    return {"intent": intent, "noise_reason": result.get("noise_reason") if noise else None, "summary": result.get("summary")}


def compare_call(call: dict, sravaani: dict, groq_text: str, config: dict) -> dict:
    """Both routes for one call, judged by the same prompt."""
    context = config.get("call_context", "a customer-service phone call")
    intents = settings.intents_for(call, config)
    row = {
        "call_uuid": call["call_uuid"],
        "client": call.get("client", "default"),
        "state_group": call["state_group"],
        "groq_text": groq_text,
        "sravaani_native": sravaani.get("text"),
        "audio_sec": sravaani.get("audio_sec"),
        "infer_sec": sravaani.get("infer_sec"),
    }
    try:
        row["sravaani_english"] = translate(sravaani["text"], settings.language_for(call, config), context) if sravaani.get("text") else ""
    except Exception as e:
        row["sravaani_english"] = ""
        row["translate_error"] = f"{type(e).__name__}: {e}"[:200]
    row["groq"] = judge(groq_text, context, intents)
    row["sravaani"] = judge(row["sravaani_english"], context, intents)
    return row


def baseline_texts(manifest: dict) -> dict:
    """Groq English per call from data/groq_out.jsonl, falling back to a transcript stored in the manifest."""
    texts = {u: m.get("transcript") for u, m in manifest.items() if m.get("transcript")}
    texts.update({r["call_uuid"]: r["text"] for r in read_jsonl(settings.BASELINE) if r.get("text") is not None})
    return texts


def main() -> None:
    """Append one comparison row per call that has both transcripts to data/compare_out.jsonl."""
    settings.require_groq()
    config = settings.load_config()
    manifest = {m["call_uuid"]: m for m in read_json(settings.MANIFEST)}
    transcripts = {r["call_uuid"]: r for r in read_jsonl(settings.TRANSCRIPTS)}
    groq_texts = baseline_texts(manifest)
    done = done_ids(settings.COMPARISON)
    todo = [u for u in manifest if u not in done and u in transcripts and u in groq_texts]
    waiting = len(manifest) - len(done) - len(todo)
    print(f"{len(todo)} to compare ({waiting} still missing a SraVaani or Groq transcript)", flush=True)
    with open(settings.COMPARISON, "a", encoding="utf-8") as out, ThreadPoolExecutor(settings.GROQ_WORKERS) as pool:
        rows = pool.map(lambda u: compare_call(manifest[u], transcripts[u], groq_texts[u], config), todo)
        for k, row in enumerate(rows, 1):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{k}/{len(todo)} groq={row['groq'].get('intent')} sravaani={row['sravaani'].get('intent')}", flush=True)


if __name__ == "__main__":
    main()
