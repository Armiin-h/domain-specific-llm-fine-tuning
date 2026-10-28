"""Score held-out clause prompts with the base model and an adapter."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from clausedraft.config import REPO_ROOT, load_config
from clausedraft.eval.decode import DecodeSettings
from clausedraft.eval.gallery import render_gallery
from clausedraft.eval.generate import generate_continuation, gold_nll, load_model, unload
from clausedraft.eval.metrics import Comparison, score_continuation, summarize
from clausedraft.eval.prompts import select_fixed_prompts


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare base and adapter continuations on held-out clauses.")
    parser.add_argument("--adapter", default=None, help="Adapter directory. Omit to score the base model only.")
    parser.add_argument("--model", default=None, help="Base model id. Defaults to the config model.")
    parser.add_argument("--output", default="docs/gallery.md", help="Gallery markdown path.")
    parser.add_argument("--report", default="outputs/eval/report.json", help="JSON metrics path.")
    args = parser.parse_args()

    config = load_config()
    model_name = args.model or config.model.name
    test_path = REPO_ROOT / config.paths.test_jsonl
    prompts = select_fixed_prompts(test_path)
    if not prompts:
        print(f"no fixed prompts in {test_path}")
        return 1

    decode = DecodeSettings()
    adapter_dir = _adapter_dir(args.adapter)
    print(f"prompts: {len(prompts)}")
    print("scoring base")
    base_rows = _score(model_name, config.quantization, None, prompts, decode)
    adapter_rows = None
    adapter_label = None
    if adapter_dir is not None:
        print(f"scoring adapter {adapter_dir}")
        adapter_rows = _score(model_name, config.quantization, adapter_dir, prompts, decode)
        adapter_label = _display_path(adapter_dir)

    comparisons = [
        Comparison(
            clause_type=example.clause_type,
            prefix=example.prefix,
            gold=example.tail,
            base=base_rows[index],
            adapter=None if adapter_rows is None else adapter_rows[index],
        )
        for index, example in enumerate(prompts)
    ]
    gallery = render_gallery(
        comparisons,
        model_name=model_name,
        adapter=adapter_label,
        decode=decode,
        note=_adapter_note(adapter_dir),
    )
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(gallery, encoding="utf-8")
    report_path = _repo_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(comparisons)
    report_path.write_text(
        json.dumps({"model": model_name, "adapter": adapter_label, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"gallery: {output}")
    for key in ("base_word_f1", "adapter_word_f1", "base_in_register", "adapter_in_register", "base_nll", "adapter_nll"):
        print(f"{key}: {summary[key]}")
    return 0


def _score(model_name, quant, adapter_dir, prompts, decode):
    model, tokenizer = load_model(model_name, quant, adapter_dir)
    scored = []
    try:
        for example in prompts:
            text = generate_continuation(model, tokenizer, example, decode)
            nll = gold_nll(model, tokenizer, example)
            scored.append(score_continuation(text, example.tail, nll))
            print(f"  {example.clause_type}: f1={scored[-1].word_f1:.3f} nll={nll:.3f}")
    finally:
        unload(model)
    return scored


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _adapter_note(adapter_dir: Path | None) -> str | None:
    if adapter_dir is not None and adapter_dir.name == "smoke":
        return (
            "The adapter scored here is the one-step smoke check, not the full epoch. "
            "Pass --adapter with the finished adapter directory to replace this gallery."
        )
    return None


def _adapter_dir(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not (path / "adapter_config.json").is_file():
        raise SystemExit(f"missing adapter_config.json in {path}")
    return path


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
