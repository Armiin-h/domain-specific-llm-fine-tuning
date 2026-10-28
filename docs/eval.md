# Evaluation

`scripts/eval_generate.py` continues a fixed set of held-out clause prefixes with the base model, then with the adapter, and writes a side-by-side gallery. The two models are loaded one at a time. An 8 GB GPU cannot hold both.

The prompts are the first test-split row of eight clause types: Governing Laws, Indemnifications, Confidentiality, Notices, Severability, Assignments, Waivers, and Terminations. The official test file is not used for training. Decoding is greedy and capped at 128 new tokens, so the only difference between the two columns is the adapter.

Each continuation is scored three ways:

- Word F1 against the gold tail.
- A register check. The continuation has to be long enough, must not open like a chat reply, and must use ordinary contract wording (`shall`, `agreement`, `party`, `herein`, and similar).
- Mean token loss of the gold tail. Prompt tokens are masked, which is the same idea as assistant-only loss during training.

```bash
python scripts/eval_generate.py --adapter outputs/adapter --output docs/gallery.md
```

`outputs/adapter` is the finished adapter. It is not in git. If that directory has no `adapter_config.json`, pass the directory that does. Omit `--adapter` to score only the base model.

The gallery is [gallery.md](gallery.md). The numeric summary is also written to `outputs/eval/report.json`.
