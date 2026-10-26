"""Run QLoRA SFT from configs/default.yaml."""

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from clausedraft.config import REPO_ROOT, load_config
from clausedraft.train.loop import resolve_output_dir, run_sft

# Qwen2.5-3B-Instruct is about 6 GB of weights. Leave headroom for the tokenizer files.
_MIN_FREE_BYTES = 8 * 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA supervised fine-tune on clause chat JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Use the first N training rows.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps.")
    parser.add_argument("--output-dir", default=None, help="Adapter directory. Defaults to the config path.")
    parser.add_argument("--model", default=None, help="Base model id. Defaults to the config model.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the last checkpoint in the output directory.",
    )
    args = parser.parse_args()

    config = load_config()
    train_path = REPO_ROOT / config.paths.train_jsonl
    eval_path = REPO_ROOT / config.paths.val_jsonl
    if not train_path.is_file():
        print(f"missing {train_path}. Run scripts/build_jsonl.py first.")
        return 1

    cache_problem = _cache_space_message()
    if cache_problem:
        print(cache_problem)
        return 1

    output_dir = resolve_output_dir(config, args.output_dir)
    result = run_sft(
        config,
        train_path=train_path,
        eval_path=eval_path if eval_path.is_file() else None,
        output_dir=output_dir,
        model_name=args.model,
        limit=args.limit,
        max_steps=args.max_steps,
        resume=args.resume,
    )
    print(f"train_rows: {result.train_rows}")
    print(f"eval_rows: {result.eval_rows}")
    for key in ("train_loss", "eval_loss", "train_runtime"):
        if key in result.metrics:
            print(f"{key}: {result.metrics[key]}")
    print(f"adapter: {result.output_dir}")
    return 0


def _cache_space_message() -> str | None:
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free
    if free >= _MIN_FREE_BYTES:
        return None
    free_gb = free / 1024**3
    return (
        f"Hugging Face cache at {cache} has {free_gb:.1f} GB free. "
        "The 3B base model needs about 8 GB. Set HF_HUB_CACHE to a directory on a larger drive and rerun."
    )


if __name__ == "__main__":
    raise SystemExit(main())
