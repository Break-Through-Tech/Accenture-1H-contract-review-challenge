# Candidate clause evidence

This package loads predictions and extracts reviewable candidate passages. It
also supports provisional clause scoring. It also aggregates contract triage scores, without training models. The category rubric is unchanged.

Run from the repository root:

```python
from risk_scoring.io import load_prediction_chunks
from risk_scoring.evidence import extract_evidence

chunks, categories = load_prediction_chunks()
evidence = extract_evidence(chunks, categories)
```

The functions return data in memory and do not write outputs. Alternate paths
are accepted by the loader. `extract_evidence` accepts either one threshold or
a complete category-to-threshold mapping, plus threshold status and provenance.
The default is 0.5, as used in the training notebook; it is marked provisional
because no independently verified threshold artifact is supplied. Never tune
these thresholds or localization cues on the test data.

`contract_id` is the original `title`, not a newly inferred identifier. Offsets
are half-open original-source character positions. Text is never normalized.
Ground-truth columns are excluded when reading the chunk Parquet file.
Probabilities describe category presence in supporting chunks, not risk or
calibrated confidence in an extracted span. The maximum is descriptive only.

Localization uses category-specific regular expressions and simple sentence /
blank-line paragraph boundaries. A following sentence containing a possible
qualification or reference is retained. These are candidate evidence spans,
not verified clause boundaries: every result requires review. Negation does
not automatically erase evidence; the full wording is preserved for review.
Abbreviations, headings, lists, distant qualifications, cross-references, and
multiple clauses in a sentence can defeat these heuristics. Broad cues may
match unrelated language. Validate and refine on development material only.

Positive-overlap spans are consolidated within contract, category, and
localization type. Source overlaps must agree character for character.
Separate occurrences, even with identical text, remain separate. Overlap chains
may combine distinct nearby provisions and are flagged when boundaries differ.
Fallback regions are deliberately not merged into localized findings: their
relationship is unknown. A fallback overlapping a localized finding is marked
as potentially duplicate and must not be counted separately without review.
Different categories can describe the same provision; cross-category grouping
and any counting policy remain future work. Evidence records are not clause
counts.

Tests use synthetic fixtures (not human risk labels), plus a real-input schema
and join check:

```sh
python3 -B -m unittest discover -s tests -p 'test_risk_evidence.py' -v
```

## Clause scoring

```python
from risk_scoring.scoring import score_clauses

scored = score_clauses(evidence)
```

Scores start from the category profile. Narrow, category-specific phrase rules
set individual signals to 0–2; their sum gives Low (0–2), Medium (3–5), or High
(6–8). Detection probabilities never affect these scores. Each result retains
its evidence and adds signal sources, matched rules, quotes, offsets, and versions.

Rules currently cover scope for 34 substantive categories, with additional
consequence, reciprocity, and exposure rules for selected categories. Informational
categories and the competitive restriction exception retain their defaults.
Unmatched signals retain unverified defaults, not inferred zero risk.

Negation, conditions, and references defer rules in the affected sentence and
adjacent qualifications. Unrelated sentences do not block all adjustments. Conflicting rules retain the default for that signal. Fallback
chunks also retain defaults. These provisional totals are not clearance decisions;
all results require review. The patterns are intentionally narrow and do not
resolve legal meaning, synonyms, or distant qualifications. Broader coverage
needs development examples and advisor review, never test-set tuning.

Run both test modules with:

```sh
python3 -B -m unittest discover -s tests -p 'test_risk*.py' -v
```

## Contract triage

```python
from risk_scoring.aggregation import aggregate_contracts

contracts = aggregate_contracts(
    scored,
    contract_ids=chunks["title"].unique().tolist(),
    review_flags={},
)
```

Pass all contract titles so contracts without detections appear as Review Required.
Optional review flags map contract titles to lists of notes, such as a missing
category needing review. These notes never change the numeric score.

Overlapping localized passages form one evidence group across categories. Each
group takes its highest clause score; the contract takes its highest group score.
The output includes High/Medium/Low group counts, source ranges, member indices,
and the original clause findings with explanations. Indices refer to the
contract's `clause_findings` list. Duplicate records remain traceable but do not
increase counts. Probabilities do not affect aggregation.

Fallback chunks and malformed scores/spans are excluded and flagged, but retained
in `clause_findings`. A contract with no usable findings has a null triage score
and Review Required level. Localized provisional scores remain eligible even
when their underlying signals need review; all contract results remain provisional.

Overlap grouping can combine nearby distinct provisions through a chain of
spans. Such groups are flagged. Identical wording at different source locations
is kept separate; semantic deduplication is not implemented. Counts describe
provisional evidence groups, not verified legal clauses. Contract scores measure
the highest detected concern, not cumulative exposure or confirmed safety.

## Reproducible pipeline

From the repository root, run a synthetic check:

```sh
python3 -B -m risk_scoring.run --synthetic --output-dir results/risk_scoring/synthetic-v0.1-review
```

Use a new folder name for each run; existing outputs are never overwritten.
The synthetic option also saves its input Parquet files in `synthetic_inputs/`.
These invented examples test execution, not risk accuracy or model performance.

For permitted development inputs, provide both paths explicitly:

```sh
python3 -B -m risk_scoring.run --predictions PATH_TO_PREDICTIONS.parquet --chunks PATH_TO_MATCHING_CHUNKS.parquet --split development --threshold 0.5 --output-dir results/risk_scoring/development-v0.1-review
```

Validation inputs must have verified `(title, chunk_id)` alignment before use;
the pipeline does not infer row identifiers for a probability-only array. The
known final test input paths are blocked pending configuration review and freeze.
The split argument records provenance; it cannot verify an arbitrary file's origin.

Outputs:

- `clause_risks.parquet`: readable score/evidence columns, group linkage, and
  `finding_json` containing the full original record, probabilities, rules, and explanations.
- `contract_risks.csv`: one row per contract, score, level, group counts, and
  `evidence_groups_json` for tracing the aggregation.
- `coverage_flags.csv`: finding uncertainties and category non-detections. A
  non-detection is not proof of absence or a risk penalty; applicability is unknown.
- `run_manifest.json`: paths and hashes, profile/rule/cue versions, all detection
  thresholds, source hashes, package versions, UTC timestamps, and result counts.

The default threshold is explicitly provisional. No test tuning occurs. Read
JSON cells with `json.loads`. A blank contract score means Review Required.
All 41 categories receive a non-detection flag when below threshold, so coverage
output can be verbose until an advisor-approved applicability policy exists.

## Verified validation run

```sh
python3 -B -m risk_scoring.run --validation --output-dir results/risk_scoring/validation-v0.1-review
```

The final notebook builds validation rows by filtering the training Parquet with
`val_titles`, preserving file order. Full mode has no subset shuffle; tokenization
maps rows in place, and `Trainer.predict` returns predictions in dataset order.
The label vector and model label mapping use `categories.json` order.

Before export, the validation loader checks full-run metadata, titles, unique keys,
probability validity, category order, positive counts, and all 41 saved validation
AP values. The current reconstruction matches AP within 1.12e-16. Matching shape
alone is insufficient. Category labels are used only for this alignment audit,
then excluded from the exported scoring inputs. No risk labels are involved.

The manifest records the audit, ordered-key hash, source hashes, and notebook hash.
Keyed validation inputs are saved in `validation_inputs/`. The original NPY lacks
row IDs and historical source hashes, so the reconstruction is corroborated by
code and saved metrics rather than cryptographically proven from the original run.
Changes to the training workflow require reviewing this reconstruction again.

The initial run processed 61 contracts and 2,893 findings: 3 Low, 57 Medium,
0 High, and 1 Review Required contract. All findings require review; 307 are
fallback chunks and 1,754 have unresolved wording markers. No wording adjustments
applied, so these results reflect profile defaults, not successful wording-based
risk discrimination. Rules and thresholds were not changed after this inspection.

Run all tests, including alignment checks:

```sh
python3 -B -m unittest discover -s tests -v
```

## Wording-rule correction (v0.2)

The v0.1 validation run had zero matched rules, not merely zero applied changes.
Most patterns were exact phrases from synthetic examples. Ordinary contract
wording and whitespace variation did not match. The passage-wide uncertainty
guard and heading-only extraction were additional limitations.

Version 0.2 adds targeted patterns for explicit grants, frequency limits, caps,
assignment consequences, and consent discretion. Matches preserve original
quotations and offsets. Permission wording (`may`) alone is not uncertainty;
negation and exceptions still defer affected rules. Positive limit constructions
such as `shall not exceed` are recognized without ignoring surrounding negation.
Short category headings retain their following provision. The scale and buckets
are unchanged, and unmatched signals retain provisional defaults.

```sh
python3 -B -m risk_scoring.run --validation --output-dir results/risk_scoring/validation-v0.2-wording-review
```

This development run produced 2,824 findings for 61 contracts. Rules applied to
15 findings, changing a signal in 12; three applications confirmed defaults.
Contract counts remained 3 Low, 57 Medium, 0 High, and 1 Review Required. All
findings still require review. The reduced finding count reflects heading/context
consolidation. Prior outputs were retained. These validation excerpts now inform
development rules and tests; this is not an independent risk-quality evaluation.
Frequency, geography, and assignment-consequence interpretations remain rubric
choices for advisor review, not legal conclusions or human annotations.

## Evidence eligibility and review flags (v0.3)

`usable_for_scoring` separates localized provisions from heading-only, contents,
and fallback evidence. Review-only findings retain text, probabilities, and
provisional scores for inspection but cannot affect contract aggregation. Heading
recovery can read up to 2,000 additional source characters and inspect four
following segments; added context does not acquire a new model probability.
Source-code escrow now requires both source-code and escrow wording. Party and
document-name localization cues are narrower.

`needs_review` now reflects concrete `review_reasons`. Ordinary limitations such
as unmatched rules and provisional defaults live in `limitations`. A false review
flag never means human approval: `risk_status` remains provisional and unreviewed.
This supersedes earlier statements that every finding requires review.

```sh
python3 -B -m risk_scoring.run --validation --output-dir results/risk_scoring/validation-v0.3-evidence-review
```

The run produced 2,408 findings: 2,008 usable and 400 excluded. There are 1,795
findings with concrete review flags and 613 without. Contract counts remain
3 Low, 57 Medium, 0 High, and 1 Review Required. The prior IP and transition
headings now include their provisions; the merchandise i-Escrow example and
party table-of-contents example remain review-only. Prior outputs are unchanged.
Heading and contents recognition remain heuristic; unresolved context and
exceptions remain flagged rather than treated as resolved or human-validated.

## Contextual defaults and substantive triage (v0.4)

Informational-only groups remain in group counts but cannot establish a contract
triage score. Contracts without usable substantive evidence are Review Required.
Unsupported Uncapped Liability findings and incidental insurance mentions are
excluded from triage while retaining their provisional scores and explanations.

Annual audit frequency and assignment-void wording are context observations, not
numeric adjustments. Worldwide/perpetual scope rules defer when the passage also
contains explicit use or duration restrictions. New narrow rules address IP scope,
insurance obligations, commitment bounds, and sharing scope or explicit discretion.
They do not infer business materiality from a payment amount or a category name.

`signal_sources` distinguishes unverified defaults from wording support. Unknown
balance remains explicitly unverified even when its numeric placeholder is zero.
Financial uncertainty is flagged separately; bounded payments do not establish
materiality. No default is a human assessment or a legal conclusion.

```sh
python3 -B -m risk_scoring.run --validation --output-dir results/risk_scoring/validation-v0.4-context-review-final
```

The final run has 2,408 findings, 1,897 included in aggregation, and 4 wording-based
score changes (9 findings with applied rules). Contracts: 0 Low, 57 Medium, 0 High,
4 Review Required. There are 1,910 findings requiring review. Contextual coverage
remains narrow and most numeric signals are still provisional defaults.
The earlier `validation-v0.4-context-review` output is retained as an intermediate
run; use the `-final` folder, which corrects the Purchase Order/insurance mismatch.

## Audit fixes (v0.5)

Heading recovery stops at another heading and retains immediately following
qualifications. Truncated recovery and incomplete chunk context are excluded
from numeric triage. Extraction and scoring share abbreviation-aware sentence
boundaries. Negated insurance requirements and discussion/question/hypothetical
wording cannot establish positive obligations. Unresolved wording defers numeric
triage; original provisional defaults remain available for inspection, not as
approved risk assessments.

Clause exports now expose `risk_status` and `numeric_score_status` directly:

- `included_in_risk_groups`: belongs to a usable evidence group, including informational groups.
- `eligible_for_primary_triage`: usable substantive evidence.
- `included_in_triage`: an eligible finding whose score equals its contract's primary maximum (ties included).
- `numeric_score_status=deferred_review`: numeric values are retained for inspection but excluded from triage.

```sh
python3 -B -m risk_scoring.run --validation --output-dir results/risk_scoring/validation-v0.5-audit-fixes
```

This run produced 2,477 findings across 61 contracts: 605 grouped findings,
438 eligible substantive findings, and 337 primary-score contributors. There
are 2,000 findings requiring review; all 61 contracts have at least one review
issue. Distribution: 0 Low, 53 Medium, 0 High, 8 Review Required. Deferred scores
remain in clause exports, so an unfiltered clause histogram is not a histogram
of usable risk evidence. All 79 tests pass, including the six audit regressions.

The sentence and operative-language checks remain conservative heuristics, not
complete language understanding. Abbreviation coverage is finite, recovery is
bounded, and unresolved references can exclude otherwise useful evidence.
Previous output directories are unchanged.

## Targeted manual-review fixes (v0.6)

The assignment heading “Additional Action to Assign Interest” now recovers its
following provision or remains excluded. Worldwide-grant checks recognize
“soley,” prior written consent, and agreement-term restrictions. A representation
of being fully licensed/registered is excluded as License Grant evidence unless
an actual unnegated grant is present. Original text and offsets are preserved.

```sh
python3 -B -m risk_scoring.run --validation --output-dir results/risk_scoring/validation-v0.6-manual-review-fixes
```

All 86 tests pass. Validation has 2,476 findings, with 2,002 requiring review.
Contract scores and levels are unchanged: 0 Low, 53 Medium, 0 High, and 8 Review
Required. PreCheck retains one substantive Medium group instead of two. The
three reviewed passages no longer contribute unsupported evidence to triage;
other valid findings preserve their contracts' primary scores. This remains
validation-driven development, not an independent quality estimate.

## Focused Audit Rights rules (v0.7)

Audit rules now recognize relevant-record limits, all-financial-record access,
explicit mutual/exclusive permission, counterparty cost shifting, and audit-related
service suspension or termination. Cost allocation affects Imbalance, not assumed
financial magnitude. Own-expense wording is recorded as context only; unknown
Exposure and missing reciprocity remain unverified. Existing uncertainty guards
remain in force, including a purpose-limit check on broad-record access.

The two natural audit examples score 1/0/0/1 = 2 (Low) and 2/2/1/1 = 6 (High).
These are provisional heuristic results, not advisor risk labels. The narrow
example still flags that cost magnitude cannot be established from allocation.

```sh
python3 -B -m risk_scoring.run --validation --output-dir results/risk_scoring/validation-v0.7-audit-context
```

All 95 tests pass. Validation contract counts remain 0 Low, 53 Medium, 0 High,
and 8 Review Required. Across 2,476 findings, applied wording-rule coverage rises
from 6 to 7 findings and actual score changes from 2 to 3. Audit Rights accounts
for the additional change. No profiles, thresholds, or aggregation rules changed.
