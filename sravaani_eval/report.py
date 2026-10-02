"""Step 4: summarise the comparison into outputs/analysis.txt and write side-by-side examples to outputs/examples.txt."""
import sys
from collections import Counter, defaultdict

from sravaani_eval import settings
from sravaani_eval.store import read_jsonl

def is_noise(result: dict) -> bool:
    """True when the analysis routed the call to NOISE."""
    return result.get("intent") == "NOISE"


def is_real(result: dict) -> bool:
    """True when the analysis produced an intent other than NOISE."""
    return bool(result.get("intent")) and not is_noise(result)


def base_intent(intent) -> str:
    """Intent without a sub-type suffix such as ' - GL'."""
    return (intent or "").rsplit(" - ", 1)[0]


def one_line(text, limit: int) -> str:
    """Text cut to a length and flattened onto one line."""
    return (text or "")[:limit].replace("\n", " ")


def median(values: list) -> int:
    """Upper median of a list of numbers."""
    return sorted(values)[len(values) // 2]


def group_summary(name: str, rows: list) -> str:
    """One pipe-separated line of counts for a group of calls."""
    both_real = [r for r in rows if is_real(r["groq"]) and is_real(r["sravaani"])]
    fields = [
        name,
        len(rows),
        sum(is_noise(r["groq"]) for r in rows),
        sum(is_noise(r["sravaani"]) for r in rows),
        len(both_real),
        sum(base_intent(r["groq"]["intent"]) == base_intent(r["sravaani"]["intent"]) for r in both_real),
        sum(is_noise(r["groq"]) and is_real(r["sravaani"]) for r in rows),
        sum(is_noise(r["sravaani"]) and is_real(r["groq"]) for r in rows),
        f"{sum(bool(r['groq'].get('error')) for r in rows)}/{sum(bool(r['sravaani'].get('error')) for r in rows)}",
    ]
    return " | ".join(str(f) for f in fields)


def analysis_lines(rows: list) -> list:
    """Every line of the analysis report."""
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r['client']} / {r['state_group']}"].append(r)
    groups["ALL"] = rows

    lines = [
        f"calls: {len(rows)}",
        "group | n | groq NOISE | sravaani NOISE | both real | same intent (both real) | groq->NOISE only "
        "| sravaani->NOISE only | errors g/s",
    ]
    lines += [group_summary(name, group) for name, group in groups.items()]

    audio = sum(r.get("audio_sec") or 0 for r in rows)
    infer = sum(r.get("infer_sec") or 0 for r in rows)
    groq_words = [len((r["groq_text"] or "").split()) for r in rows]
    sravaani_words = [len((r["sravaani_english"] or "").split()) for r in rows]
    sravaani_errors = Counter(r["sravaani"].get("error", "")[:60] for r in rows if r["sravaani"].get("error"))
    groq_errors = Counter(r["groq"].get("error", "")[:60] for r in rows if r["groq"].get("error"))
    lines += [
        f"\nspeed: {audio / 3600:.2f} h audio in {infer / 60:.1f} min CPU (ratio {infer / audio if audio else 0:.2f})",
        f"english words per call median: groq {median(groq_words)} sravaani {median(sravaani_words)}",
        f"errors: {sravaani_errors} {groq_errors}",
        f"translate errors: {sum(1 for r in rows if r.get('translate_error'))}",
        "\n===== DISAGREEMENTS (NOISE vs real)",
    ]
    for r in rows:
        g, s = r["groq"], r["sravaani"]
        if is_noise(g) != is_noise(s) and g.get("intent") and s.get("intent"):
            lines += [
                f"\n## {r['client']} {r['state_group']} {r['call_uuid'][:8]} | groq={g['intent']} ({g.get('noise_reason')}) "
                f"| sravaani={s['intent']} ({s.get('noise_reason')})",
                f"GROQ    : {one_line(r['groq_text'], 420)}",
                f"SRAVAANI: {one_line(r['sravaani_english'], 420)}",
            ]
    lines.append("\n===== INTENT DIFFERENCES (both real)")
    for r in rows:
        g, s = r["groq"], r["sravaani"]
        if is_real(g) and is_real(s) and base_intent(g["intent"]) != base_intent(s["intent"]):
            lines.append(
                f"- {r['client']} {r['state_group']} {r['call_uuid'][:8]} groq={g['intent']} | sravaani={s['intent']} "
                f"| S: {(r['sravaani_english'] or '')[:200]}"
            )
    return lines


def example_lines(rows: list, prefixes: list) -> list:
    """Groq English, SraVaani native and SraVaani English for each chosen call."""
    by_prefix = {r["call_uuid"][:8]: r for r in rows}
    lines = []
    for prefix in prefixes:
        r = by_prefix.get(prefix[:8])
        if r is None:
            print(f"  no comparison row for {prefix}; skipped")
            continue
        lines += [
            f"\n##### {prefix} {r['client']} {r['state_group']} audio={r.get('audio_sec')}s "
            f"groq={r['groq'].get('intent')} sravaani={r['sravaani'].get('intent')} "
            f"summary: {r['sravaani'].get('summary')}",
            f"GROQ: {one_line(r['groq_text'], 600)}",
            f"NATIVE: {one_line(r['sravaani_native'], 300)}",
            f"SV-EN: {one_line(r['sravaani_english'], 600)}",
        ]
    return lines


def write_lines(path, lines: list) -> None:
    """Write lines to a UTF-8 text file, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Write both reports; call-id prefixes given on the command line replace the config's examples."""
    rows = read_jsonl(settings.COMPARISON)
    if not rows:
        raise SystemExit(f"No comparison rows in {settings.COMPARISON}; run the compare step first.")
    write_lines(settings.ANALYSIS, analysis_lines(rows))
    write_lines(settings.EXAMPLES, example_lines(rows, sys.argv[1:] or settings.load_config().get("examples", [])))
    print(f"wrote {settings.ANALYSIS} and {settings.EXAMPLES}")


if __name__ == "__main__":
    main()
