"""Print a few chat examples from the written JSONL files."""

import json

from clausedraft.config import REPO_ROOT, load_config
from clausedraft.data.messages import render_messages


def main() -> int:
    config = load_config()
    path = REPO_ROOT / config.paths.train_jsonl
    if not path.is_file():
        print(f"missing {path}. Run scripts/build_jsonl.py first.")
        return 1
    shown = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            print(render_messages(json.loads(line), shown + 1))
            print()
            shown += 1
            if shown == 3:
                break
    if shown == 0:
        print(f"{path} has no examples")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
