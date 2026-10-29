"""Continue one clause from a type and a prefix."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from clausedraft.config import REPO_ROOT, load_config
from clausedraft.infer.engine import Drafter
from clausedraft.infer.types import CLAUSE_TYPES


def main() -> int:
    parser = argparse.ArgumentParser(description="Continue a commercial-contract clause.")
    parser.add_argument("--type", default=None, help="LEDGAR clause type, such as Governing Laws.")
    parser.add_argument("--prefix", default=None, help="The start of the clause.")
    parser.add_argument("--preset", default="greedy", help="greedy or draft.")
    parser.add_argument("--adapter", default=None, help="Adapter directory. Omit to use the base model only.")
    parser.add_argument("--model", default=None, help="Base model id. Defaults to the config model.")
    parser.add_argument("--list-types", action="store_true", help="Print the clause types and exit.")
    parser.add_argument("--health", action="store_true", help="Load the model and print readiness.")
    args = parser.parse_args()

    if args.list_types:
        for name in CLAUSE_TYPES:
            print(name)
        return 0

    adapter_dir = _adapter_dir(args.adapter)
    drafter = Drafter.load(load_config(), adapter_dir, args.model)
    status = drafter.health()
    print(f"ok: {status.ok}")
    print(f"cuda: {status.cuda}")
    print(f"adapter: {status.adapter}")
    print(f"clause_types: {status.clause_types}")
    if args.health:
        return 0
    if not args.type or args.prefix is None:
        print("Pass --type and --prefix, or --health, or --list-types.")
        return 1
    result = drafter.complete(args.type, args.prefix, args.preset)
    print(f"preset: {result.preset}")
    print(f"clause_type: {result.clause_type}")
    print(result.continuation)
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
