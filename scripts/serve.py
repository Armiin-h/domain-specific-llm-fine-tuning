"""Serve the completion API and the page."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import uvicorn

from clausedraft.api.app import create_app
from clausedraft.api.fake import EchoDrafter
from clausedraft.config import REPO_ROOT, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the ClauseDraft page and completion API.")
    parser.add_argument("--adapter", default=None, help="Adapter directory. Omit to use the base model only.")
    parser.add_argument("--model", default=None, help="Base model id. Defaults to the config model.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Skip the model and return a fixed continuation. Useful for checking the page.",
    )
    args = parser.parse_args()
    if args.fake:
        drafter = EchoDrafter()
    else:
        from clausedraft.infer.engine import Drafter

        drafter = Drafter.load(load_config(), _adapter_dir(args.adapter), args.model)
    uvicorn.run(create_app(drafter), host=args.host, port=args.port)
    return 0


def _adapter_dir(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not (path / "adapter_config.json").is_file():
        raise SystemExit(f"missing adapter_config.json in {path}")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
