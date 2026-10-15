"""Print length and label stats for the raw LEDGAR download."""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from clausedraft.config import load_config
from clausedraft.data.corpus import load_ledgar
from clausedraft.data.stats import format_stats, summarize


def main() -> int:
    config = load_config()
    provisions = load_ledgar(config.data.dataset, config.data.subset)
    stats = summarize(
        provisions,
        min_chars=config.data.min_chars,
        max_chars=config.data.max_chars,
    )
    print(format_stats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
