# ClauseDraft

Fine-tune a small open instruct model on commercial contract clauses, then continue a partial clause in that register.

The base model is [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct). Training is QLoRA: the base weights stay frozen in 4-bit NF4, and a LoRA adapter learns the clause style. The corpus is the LEDGAR subset of [LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue) — contract provisions from public SEC filings, labeled with 100 clause types. Provisions are public filings; the LEDGAR annotations are research-only (CC BY-NC 4.0), so the corpus is downloaded at runtime and is not stored in git.

Outputs are drafting demos, not legal advice.

## What is in the tree

The package loads `configs/default.yaml`, probes the local GPU for a working 4-bit kernel, and prepares the LEDGAR corpus. `scripts/prepare_corpus.py` downloads the provisions, drops headings, exact copies, and near-copies, caps the train split at 12,000 stratified rows, and writes `data/processed/provisions.jsonl` (gitignored). Chat-format training files, the trainer, evaluation, and the query API are not in this tree yet.

## Stack

| Layer | Choice |
| --- | --- |
| Base model | Qwen2.5-3B-Instruct, fallback Qwen2.5-1.5B-Instruct |
| Training | Hugging Face Transformers, PEFT, TRL (`SFTTrainer`) |
| Quantization | bitsandbytes NF4, double quant, bfloat16 compute |
| Corpus | `coastalcph/lex_glue`, config `ledgar` |
| Hardware target | 8 GB NVIDIA GPU (tested on an RTX 5060 Laptop) |

## Setup

Python 3.11+ and an NVIDIA driver that can run CUDA 13.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/check_gpu.py
python scripts/inspect_corpus.py
python scripts/prepare_corpus.py
pytest
```

`requirements.txt` pulls `torch==2.14.1+cu130` from the PyTorch CUDA 13.0 index. The wheel published on PyPI under the same version number is the CPU build, which will not run 4-bit training on this GPU.

## Layout

```text
configs/default.yaml       training hyperparameters
src/clausedraft/           package
src/clausedraft/data/      LEDGAR load, clean, dedupe, sample
scripts/check_gpu.py       CUDA and NF4 smoke check
scripts/inspect_corpus.py  raw length and label counts
scripts/prepare_corpus.py  cleaned provisions JSONL
tests/                     unit tests
docs/architecture.md       data flow
docs/data.md               corpus, license, filters
```

See [docs/architecture.md](docs/architecture.md) and [docs/data.md](docs/data.md).
