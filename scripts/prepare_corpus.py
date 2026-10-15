"""Download LEDGAR, clean it, and write data/processed/provisions.jsonl."""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from clausedraft.config import load_config
from clausedraft.data.corpus import load_ledgar
from clausedraft.data.prepare import PROCESSED_PATH, format_prepare, prepare_provisions, write_provisions


def main() -> int:
    config = load_config()
    provisions = load_ledgar(config.data.dataset, config.data.subset)
    result = prepare_provisions(
        provisions,
        min_chars=config.data.min_chars,
        max_chars=config.data.max_chars,
        train_cap=config.data.train_sample_cap,
        seed=config.data.seed,
    )
    write_provisions(PROCESSED_PATH, result.provisions)
    print(format_prepare(result))
    print(f"wrote: {PROCESSED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
