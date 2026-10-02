"""Step 1: build data/manifest.json from folders of call recordings listed in eval_config.json."""
import random
from pathlib import Path

from sravaani_eval import settings
from sravaani_eval.store import write_json

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


def recordings(folder: Path) -> list:
    """Every audio file directly inside a folder, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def main() -> None:
    """Write one manifest entry per recording, optionally capped to a reproducible random subset per group."""
    config = settings.load_config()
    rnd = random.Random(config.get("seed", 42))
    manifest = []
    for group in config["groups"]:
        folder = settings.AUDIO_DIR / group["audio_subdir"]
        if not folder.is_dir():
            raise SystemExit(f"Missing audio folder {folder} for group '{group['label']}'.")
        files = recordings(folder)
        chosen = sorted(rnd.sample(files, group["calls"])) if group.get("calls") and group["calls"] < len(files) else files
        print(f"{group['label']}: {len(files)} recordings -> {len(chosen)} chosen", flush=True)
        for path in chosen:
            manifest.append({
                "call_uuid": path.stem,
                "client": group.get("client", "default"),
                "state_group": group["label"],
                "language": group["language"],
                "audio": str(path.relative_to(settings.AUDIO_DIR)).replace("\\", "/"),
            })
    write_json(settings.MANIFEST, manifest)
    print(f"manifest: {len(manifest)} calls")


if __name__ == "__main__":
    main()
