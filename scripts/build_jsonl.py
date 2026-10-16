"""Build chat JSONL files from data/processed/provisions.jsonl."""

from clausedraft.config import REPO_ROOT, load_config
from clausedraft.data.examples import build_examples, read_provisions
from clausedraft.data.export import format_counts, write_chat_splits
from clausedraft.data.leakage import assert_no_holdout_overlap, format_leakage, remove_holdout_overlap
from clausedraft.data.messages import render_messages
from clausedraft.data.prepare import PROCESSED_PATH


def main() -> int:
    if not PROCESSED_PATH.is_file():
        print(f"missing {PROCESSED_PATH}. Run scripts/prepare_corpus.py first.")
        return 1

    config = load_config()
    provisions = read_provisions(PROCESSED_PATH)
    built = build_examples(provisions, config.data.prefix_fraction)
    cleaned, report = remove_holdout_overlap(built.examples)
    assert_no_holdout_overlap(cleaned)
    counts = write_chat_splits(
        cleaned,
        root=REPO_ROOT,
        train_jsonl=config.paths.train_jsonl,
        val_jsonl=config.paths.val_jsonl,
        test_jsonl=config.paths.test_jsonl,
    )
    print(f"skipped_unsplit: {built.skipped}")
    print(format_leakage(report))
    print(format_counts(counts))
    print(f"wrote: {config.paths.train_jsonl}, {config.paths.val_jsonl}, {config.paths.test_jsonl}")
    shown = 0
    for example in cleaned:
        if example.split != "train":
            continue
        print(render_messages(example.as_json(), shown + 1))
        print()
        shown += 1
        if shown == 2:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
