# Training

`scripts/train.py` fine-tunes Qwen2.5-3B-Instruct with QLoRA on the chat JSONL from `scripts/build_jsonl.py`.

The base weights load in 4-bit NF4 with double quantization and bfloat16 compute. A LoRA adapter of rank 16 is trained on the attention and MLP projections. The optimizer is 8-bit paged AdamW. Warmup is 3 percent of the optimizer steps (`warmup_ratio` in the config; this Transformers release takes that as a step count). Loss is computed only on the assistant turn, which is the gold clause tail. On Windows the trainer uses the model's own cross-entropy, because the fused loss kernel in TRL needs Triton and that library is not in the Windows PyTorch wheel. The frozen base model is not written to disk.

## Smoke check

One optimizer step on 32 training rows, saved under `outputs/smoke` (gitignored):

```bash
python scripts/train.py --limit 32 --max-steps 1 --output-dir outputs/smoke
```

The command prints `train_loss` and writes `outputs/smoke/adapter_config.json`, the adapter weights, and `loss.json`.

The base model is downloaded into the Hugging Face hub cache on first use (about 6 GB for the 3B checkpoint). If that drive is short on space, set `HF_HUB_CACHE` to a directory elsewhere before running the command.

## Full run

```bash
python scripts/train.py
```

That uses `configs/default.yaml`: one epoch, batch size 1, gradient accumulation 8, sequence length 1024, learning rate 2e-4. Checkpoints land in `outputs/adapter`. Validation runs every 50 steps. The test split is not passed to the trainer.

If the 3B model runs out of memory, repeat the command with the smaller fallback:

```bash
python scripts/train.py --model Qwen/Qwen2.5-1.5B-Instruct
```
