# Training

`scripts/train.py` fine-tunes Qwen2.5-3B-Instruct with QLoRA on the chat JSONL from `scripts/build_jsonl.py`.

The base weights load in 4-bit NF4 with double quantization and bfloat16 compute. A LoRA adapter of rank 16 is trained on the attention and MLP projections. The optimizer is 8-bit paged AdamW. Warmup is 3 percent of the optimizer steps (`warmup_ratio` in the config; this Transformers release takes that as a step count). Loss is computed only on the assistant turn, which is the gold clause tail. On Windows the trainer uses the model's own cross-entropy, because the fused loss kernel in TRL needs Triton and that library is not in the Windows PyTorch wheel. The frozen base model is not written to disk.

## Smoke check

One optimizer step on 32 training rows, saved under `outputs/smoke` (gitignored):

```bash
python scripts/train.py --limit 32 --max-steps 1 --output-dir outputs/smoke
```

The command prints `train_loss` and writes `outputs/smoke/adapter_config.json`, the adapter weights, `loss.json`, `loss.svg`, and `model_card.md`.

The base model is downloaded into the Hugging Face hub cache on first use (about 6 GB for the 3B checkpoint). If that drive is short on space, set `HF_HUB_CACHE` to a directory elsewhere before running the command.

## Full run

The epoch does not run on the 8 GB laptop. It runs on a free Colab T4, which has 16 GB and a Linux torch build, and the adapter is stored on Google Drive so a dropped session can continue.

Open [notebooks/train_colab.ipynb](../notebooks/train_colab.ipynb) in Colab after this repo is on GitHub:

[ClauseDraft Colab notebook](https://colab.research.google.com/github/Armiin-h/domain-specific-llm-fine-tuning/blob/main/notebooks/train_colab.ipynb)

The notebook clones the repo, installs `requirements-colab.txt` (no torch pin), prepares LEDGAR, and calls `scripts/train.py`. That uses `configs/default.yaml`: one epoch, batch size 1, gradient accumulation 8, sequence length 1024, learning rate 2e-4. The test split is not passed to the trainer. Validation runs every 50 steps on the first 128 rows of the validation file. Scoring the full validation file on that cadence would take longer than the epoch.

The Drive folder receives checkpoints every 100 steps, then the adapter weights, `loss.json`, `loss.svg`, and `model_card.md`. The card records the base model id, the sha256 of the train file, the hyperparameters, and the final losses. Adapter weights stay out of git. If the session dies, rerun the resume cell in the notebook.

A checkpoint directory left on the laptop under `outputs/adapter` is an interrupted run, not a finished adapter.

If a hosted GPU runs out of memory, repeat the train command with the smaller fallback:

```bash
python scripts/train.py --model Qwen/Qwen2.5-1.5B-Instruct
```
