# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Permission Rule

**Do not write, edit, create, delete, or run anything that changes files, notebooks, git state, or GitHub (issues, PRs, branches) without the user's explicit permission first.** Reading files and explaining concepts is fine. The user implements tasks themselves; propose changes and wait for approval.

## Project

Break Through Tech AI Studio (Fall 2026) challenge for Accenture: build a contract-review triage pipeline on the CUAD dataset (510 commercial contracts, 41 clause categories). Target pipeline, per `Challenge-Project-Overview.md`:

1. **Clause detection** — chunk-based multi-label classification (TF-IDF/keyword baseline in Sept → fine-tuned transformer encoder in Oct). Evaluate with per-category precision/recall/F1; accuracy is misleading because of heavy class imbalance.
2. **Risk scoring** — explainable rule-based "four-signal" layer flagging clauses Low/Medium/High, rolled up to a contract-level triage score (Nov). No ground-truth labels: validated via Spearman correlation and bucket agreement against the advisor's hand-ranked clauses, plus sensitivity analysis of the High/Medium boundary.

Stretch goals (span extraction, trained risk model, LLM explanations, Streamlit/Gradio demo, active learning) must not displace the core deliverables.

## Environment

Work is entirely in Jupyter notebooks under `notebooks/`; there is no package, build, lint, or test suite.

```bash
python -m venv Accenture.venv && source Accenture.venv/bin/activate   # venv name is gitignored
pip install -r requirements.txt   # pandas, scikit-learn, matplotlib, ipykernel
jupyter nbconvert --to notebook --execute notebooks/<name>.ipynb --output-dir .scratch/   # headless run
```

Notebooks load data with relative paths (`../data/...`), so they must run with `notebooks/` as the working directory. Add new dependencies (e.g. transformers, torch) to `requirements.txt`.

## Data (`data/`, committed to git)

All three files are SQuAD 2.0-shaped: `data → [contract] → paragraphs → [{context, qas: [{id, question, answers[{text, answer_start}], is_impossible}]}]`. Each contract has exactly **one** paragraph whose `context` is the full contract text — documents are long, so chunking is required before any transformer.

| File | Contracts | QAs | Notes |
|---|---|---|---|
| `CUADv1.json` | 510 | 20,910 | Full dataset, 41 QAs/contract, a QA may have multiple answers |
| `train_separate_questions.json` | 408 | 22,450 | Official CUAD train split. Multi-answer QAs are **exploded into separate QAs with ≤1 answer each**, ids suffixed `_0`, `_1`, … |
| `test.json` | 102 | 4,182 | Official CUAD test split. Original format: one QA per category, possibly several answers |

- **The team uses the official split (`train_separate_questions.json` / `test.json`) for everything.** The 80/20 split in `Extract_Transform_Load.ipynb` (from `CUADv1.json`, `random_state=42`) is reference only.
- QA `id` format: `<contract title>__<Category>[_<n>]`; the clause category is also embedded in `question` (`... related to "<Category>" ...`). Use these to derive the 41 category labels.
- Because train and test are structured differently, per-contract / per-category label counts must be aggregated (dedupe by contract + category) before comparing train vs test or building multi-label targets.
- `is_impossible=True` ⇔ `answers` is empty; `answer_start` offsets index into `context` and were verified correct.

## Notebooks

- `Extract_Transform_Load.ipynb` — loads `CUADv1.json`, reference contract-level split.
- `Data_Cleaning.ipynb` — quality checks (all passed: no missing fields, duplicate ids, bad offsets, or train/test contract overlap) and flattening into DataFrames with columns `title, context, question, answer_text, answer_start, id, is_impossible` (`is_impossible` cast to int). Train → 22,450 rows, test → 5,581 rows (test multi-answer QAs exploded to one row per answer). Outputs stay in memory; nothing is written to disk.

Splits must remain at the contract level (never split chunks of one contract across train/test).
