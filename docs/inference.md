# Inference

`scripts/complete.py` loads the base model in 4-bit, attaches an adapter when one is given, and continues a single clause. The clause type has to be one of the 100 LEDGAR labels. Matching ignores case and extra spaces.

Two decoding presets are available:

- `greedy` continues up to 128 tokens with sampling off. This is the same setup as the held-out gallery.
- `draft` samples with temperature 0.7 and top-p 0.9, up to 160 tokens.

```bash
python scripts/complete.py --list-types
python scripts/complete.py --health --adapter outputs/smoke
python scripts/complete.py --adapter outputs/smoke --type "Governing Laws" --prefix "This Agreement shall be governed"
```

`--health` loads the model and prints whether CUDA is available, which adapter is attached, and how many clause types the checker knows. A completion without `--adapter` uses the base model alone.

The text is a drafting demo, not legal advice.
