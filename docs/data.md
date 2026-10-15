# Data

ClauseDraft trains on contract provisions from the LEDGAR subset of [LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue) (`coastalcph/lex_glue`, config `ledgar`).

LEDGAR is a corpus of provisions cut from US SEC EDGAR filings and labeled by clause type. LexGLUE republishes a 100-way split of that corpus:

| Split | Rows |
| --- | --- |
| train | 60,000 |
| validation | 10,000 |
| test | 10,000 |

Each row is `text` plus a `label` index. The loader maps the index to the ClassLabel name (`Governing Laws`, `Notices`, `Indemnifications`, and the other 97 types).

## Attribution

The provisions come from public SEC filings. The labels and the curated corpus are research-only.

LEDGAR: Tuggener, von Däniken, Peetz, and Cieliebak, 2020. [LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts](https://aclanthology.org/2020.lrec-1.155/). CC BY-NC 4.0.

LexGLUE: Chalkidis et al., 2022. [LexGLUE: A Benchmark Dataset for Legal Language Understanding in English](https://aclanthology.org/2022.acl-long.297/).

The text is downloaded at runtime into the Hugging Face cache. It is not committed. `data/raw/` and `data/processed/` are gitignored.

## Filters

`scripts/prepare_corpus.py` reads `configs/default.yaml` and writes `data/processed/provisions.jsonl` with `text`, `clause_type`, and `split`.

1. Collapse whitespace. The stored clause keeps its original capitalization.
2. Drop placeholder lines (`Reserved.`, `Intentionally omitted.`) and short all-caps headings.
3. Drop provisions shorter than 80 characters or longer than 4,000. The short ones are headings. The long ones do not fit a 1,024-token training sequence once the prompt is added.
4. Drop exact copies after lowercasing and stripping punctuation. If the same clause is in train and in the test split, the test copy stays and the train copy goes.
5. Inside one split and one clause type, drop a second clause when it shares a 10-word opening and the word-set overlap is at least 0.92. A one-word edit in a long standard clause is the case this catches. Different clause types are left alone.
6. Cap the train split at 12,000 rows, stratified by clause type, seed 42. Validation and test are filtered the same way and are not capped. The official test split is never used as training text.

## Commands

```bash
python scripts/inspect_corpus.py
python scripts/prepare_corpus.py
```

`inspect_corpus.py` prints raw length and label counts. `prepare_corpus.py` prints how many rows each filter removed and the final split sizes.
