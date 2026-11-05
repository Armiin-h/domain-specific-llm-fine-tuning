# Quickstart

The page and the saved comparison do not need a finished epoch. The gallery already in the repo is the one-step smoke adapter. A live continuation uses that same adapter when it is on disk. `--fake` skips the model and only checks the form.

## Page

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/serve.py --fake
```

Open `http://127.0.0.1:8000`. The form continues a clause. The section under it is the held-out base-versus-adapter gallery from [gallery.md](gallery.md).

## One real continuation

This needs the CUDA wheel from `requirements.txt`, a disk large enough for the 3B weights, and the smoke adapter.

```bash
python scripts/check_gpu.py
python scripts/train.py --limit 32 --max-steps 1 --output-dir outputs/smoke
python scripts/serve.py --adapter outputs/smoke
```

Point `HF_HUB_CACHE` at a directory on that disk before the train or serve command. The full epoch is the Colab notebook, not the laptop. See [training.md](training.md).

## Tests

```bash
pytest
```

GitHub Actions runs the same unit tests on a CPU runner. That job does not download the model and does not train. See [architecture.md](architecture.md) for the path from a labeled provision to a continuation.
