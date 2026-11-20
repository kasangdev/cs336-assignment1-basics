import json
from pathlib import Path


def save_config(config: dict, out: str | Path):
    with open(out, 'x', encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def append_metrics(record: dict, out: str | Path):
    with open(out, 'a', encoding="utf-8") as file:
        line = json.dumps(record, ensure_ascii=False)
        file.write(line + "\n")
