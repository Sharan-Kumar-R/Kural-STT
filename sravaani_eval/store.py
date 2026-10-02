"""Small JSON and JSON Lines helpers used by every step."""
import json
from pathlib import Path


def read_json(path: Path):
    """Parsed contents of a UTF-8 JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value) -> None:
    """Write a value as indented UTF-8 JSON, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1)


def read_jsonl(path: Path) -> list:
    """Every row of a JSON Lines file, or an empty list when it does not exist."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_ids(path: Path) -> set:
    """call_uuid of every row already written, so a rerun resumes."""
    return {row["call_uuid"] for row in read_jsonl(path)}
