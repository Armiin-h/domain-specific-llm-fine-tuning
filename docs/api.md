# The completion page

`scripts/serve.py` loads the base model in 4-bit, attaches an adapter when one is given, and serves a page plus a small JSON API. The page and the API share one process. Both models are never loaded together, so the page continues with the adapter (or the base model alone) and the side-by-side gallery stays a precomputed document.

```bash
python scripts/serve.py --adapter outputs/smoke
```

Open `http://127.0.0.1:8000`. Pick a clause type, paste the start of a clause, and press Continue. Four example prefixes fill the form. The line under the title stays on the page: drafting demo, not legal advice.

An empty box is rejected in the page before a request is sent. An unknown clause type or preset comes back as HTTP 400 with the same message the command-line checker uses.

`--fake` skips the model and returns a fixed continuation. It is there so the page can be clicked through without loading the 3B weights.

## Routes

| Method | Path | Body |
| --- | --- | --- |
| GET | `/api/health` | whether a model is loaded, the adapter path, CUDA, clause-type count, presets |
| GET | `/api/types` | the 100 LEDGAR clause types |
| GET | `/api/examples` | the built-in prefixes |
| POST | `/api/complete` | `clause_type`, `prefix`, optional `preset` (`greedy` or `draft`) |

A successful completion returns the canonical clause type, the stripped prefix, the continuation, and the preset that was used.

```bash
curl -s http://127.0.0.1:8000/api/complete ^
  -H "Content-Type: application/json" ^
  -d "{\"clause_type\":\"Governing Laws\",\"prefix\":\"This Agreement shall be governed\",\"preset\":\"greedy\"}"
```

The smoke adapter in `outputs/smoke` is one optimizer step. Treat a continuation from it as a wiring check, the same way the gallery does.
