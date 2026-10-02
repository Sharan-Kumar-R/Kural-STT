#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-venv/bin/python}"
LOG="data/transcribe.log"
mkdir -p data
total=$("$PY" -c "import json;from sravaani_eval import settings;print(len(json.load(open(settings.MANIFEST))))")
for i in $(seq 1 "${MAX_RUNS:-12}"); do
  echo "=== run $i" | tee -a "$LOG"
  PYTHONIOENCODING=utf-8 "$PY" -m sravaani_eval.transcribe 2>&1 | tee -a "$LOG"
  done=$("$PY" -c "from sravaani_eval import settings;from sravaani_eval.store import read_jsonl;print(len({r['call_uuid'] for r in read_jsonl(settings.TRANSCRIPTS) if r.get('text') is not None}))")
  echo "run $i finished: $done of $total calls transcribed"
  [ "$done" -ge "$total" ] && break
done
