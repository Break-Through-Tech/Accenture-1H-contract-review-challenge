"""Run evidence extraction, clause scoring, and contract triage."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import pandas as pd
import pyarrow

from .aggregation import aggregate_contracts
from .evidence import extract_evidence, DEFAULT_THRESHOLD, CUE_VERSION
from .io import load_prediction_chunks, load_categories, DEFAULT_CATEGORIES, DEFAULT_PREDICTIONS, DEFAULT_CHUNKS, ROOT, load_validation_inputs
from .scoring import score_clauses, load_profiles, PROFILE_PATH, RULE_VERSION, SIGNALS

OUTPUTS = ('clause_risks.parquet', 'contract_risks.csv', 'coverage_flags.csv', 'run_manifest.json')
CLAUSE_COLUMNS = ['contract_id', 'evidence_id', 'category', 'original_text', 'char_start', 'char_end',
                  'severity', 'scope', 'imbalance', 'exposure', 'risk_total', 'risk_level',
                  'max_detection_probability', 'localization', 'needs_review', 'finding_index',
                  'evidence_group_id', 'included_in_risk_groups', 'eligible_for_primary_triage',
                  'included_in_triage', 'risk_status', 'numeric_score_status', 'finding_json']
CONTRACT_COLUMNS = ['contract_id', 'triage_score', 'risk_level', 'high_groups', 'medium_groups',
                    'low_groups', 'evidence_group_count', 'clause_finding_count', 'excluded_finding_count',
                    'needs_review', 'risk_status', 'evidence_groups_json']
FLAG_COLUMNS = ['contract_id', 'category', 'evidence_id', 'group_id', 'flag', 'detail',
                'max_detection_probability', 'threshold']


def file_info(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def run_pipeline(predictions_path, chunks_path, output_dir, threshold=DEFAULT_THRESHOLD, data_split='development', alignment_audit=None):
    predictions_path, chunks_path, output_dir = map(Path, (predictions_path, chunks_path, output_dir))
    if data_split not in ('synthetic', 'development', 'validation'):
        raise ValueError('Only synthetic, development, or validation runs are currently enabled')
    if (predictions_path.resolve() == DEFAULT_PREDICTIONS.resolve()
            or chunks_path.resolve() == DEFAULT_CHUNKS.resolve()):
        raise ValueError('Final test inputs are blocked until configuration review and freeze')
    if any((output_dir / name).exists() for name in OUTPUTS):
        raise FileExistsError('Choose a new output directory; existing run outputs are not overwritten')
    started = datetime.now(timezone.utc).isoformat()
    chunks, categories = load_prediction_chunks(predictions_path, chunks_path)
    evidence = extract_evidence(chunks, categories, threshold=threshold,
                                threshold_source='explicit pipeline threshold; provisional, not tuned on test data')
    scored = score_clauses(evidence)
    contracts = aggregate_contracts(scored, contract_ids=chunks.title.unique())
    profiles = load_profiles()
    clause_rows, contract_rows, flags = [], [], []
    for contract in contracts:
        contract_id = contract['contract_id']
        group_ids = {index: group['group_id'] for group in contract['evidence_groups'] for index in group['finding_indices']}
        for index, finding in enumerate(contract['clause_findings']):
            row = {key: finding[key] for key in CLAUSE_COLUMNS if key in finding}
            row.update(finding['risk_scores'])
            row.update(finding_index=index, evidence_group_id=group_ids.get(index),
                       included_in_risk_groups=index in group_ids,
                       eligible_for_primary_triage=index in group_ids and finding['substantive_evidence'],
                       included_in_triage=index in group_ids and finding['substantive_evidence'] and finding['risk_total'] == contract['triage_score'],
                       finding_json=json.dumps(finding, sort_keys=True))
            clause_rows.append(row)
        counts = contract['evidence_group_counts']
        contract_rows.append({
            'contract_id': contract_id, 'triage_score': contract['triage_score'],
            'risk_level': contract['risk_level'], 'high_groups': counts['High'],
            'medium_groups': counts['Medium'], 'low_groups': counts['Low'],
            'evidence_group_count': contract['evidence_group_count'],
            'clause_finding_count': len(contract['clause_findings']),
            'excluded_finding_count': len(contract['excluded_finding_indices']),
            'needs_review': contract['needs_review'], 'risk_status': contract['risk_status'],
            'evidence_groups_json': json.dumps(contract['evidence_groups'], sort_keys=True),
        })
        for flag in contract['review_flags']:
            index = flag.get('finding_index')
            finding = contract['clause_findings'][index] if index is not None else {}
            flags.append({'contract_id': contract_id, 'category': finding.get('category'),
                          'evidence_id': finding.get('evidence_id'), 'group_id': flag.get('group_id', group_ids.get(index)),
                          'flag': flag['reason'], 'detail': 'Review flag only; no risk penalty.',
                          'max_detection_probability': finding.get('max_detection_probability'), 'threshold': threshold})
    for contract_id, group in chunks.groupby('title', sort=True):
        for category in categories:
            probability = float(group[category].max())
            if probability < threshold:
                flags.append({'contract_id': contract_id, 'category': category,
                              'flag': 'not_detected_at_threshold',
                              'detail': 'Not proof of absence; applicability unknown. Review if: ' + profiles['profiles'][category]['missing_category_review']['context'],
                              'max_detection_probability': probability, 'threshold': threshold})
    distribution = dict.fromkeys(['Low', 'Medium', 'High', 'Review Required'], 0)
    distribution.update(Counter(contract['risk_level'] for contract in contracts))
    manifest = {
        'pipeline_version': '0.1-provisional', 'data_split': data_split,
        'alignment_audit': alignment_audit,
        'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': {'predictions': file_info(predictions_path), 'chunks': file_info(chunks_path),
                   'categories': file_info(DEFAULT_CATEGORIES), 'profiles': file_info(PROFILE_PATH)},
        'configuration': {'profile_version': profiles['profile_version'], 'rule_version': RULE_VERSION,
                          'cue_version': CUE_VERSION, 'status': 'provisional_not_frozen',
                          'threshold_status': 'provisional', 'detection_thresholds': dict.fromkeys(categories, threshold),
                          'threshold_note': '0.5 is used by the training notebook; no separately verified threshold artifact is available. No tuning performed.'},
        'source_files': {name: file_info(Path(__file__).parent / name) for name in
                         ('run.py', 'io.py', 'cues.py', 'evidence.py', 'scoring.py', 'aggregation.py')},
        'environment': {'python': platform.python_version(), 'pandas': pd.__version__, 'pyarrow': pyarrow.__version__},
        'counts': {'input_chunks': len(chunks), 'contracts': len(contracts), 'clause_findings': len(scored),
                   'coverage_flags': len(flags), 'uncertain_findings': sum(bool(finding['needs_review']) for finding in scored),
                   'contract_risk_distribution': distribution,
                   'clause_risk_distribution': dict(Counter(finding['risk_level'] for finding in scored))},
        'notes': ['No ground-truth labels used for scoring.', 'All scores are provisional heuristics, not human-reviewed labels.',
                  'Detection probability is separate from risk.', 'Synthetic results verify execution, not model quality.'],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clause_rows, columns=CLAUSE_COLUMNS).to_parquet(output_dir / OUTPUTS[0], index=False)
    pd.DataFrame(contract_rows, columns=CONTRACT_COLUMNS).to_csv(output_dir / OUTPUTS[1], index=False)
    pd.DataFrame(flags, columns=FLAG_COLUMNS).to_csv(output_dir / OUTPUTS[2], index=False)
    manifest['outputs'] = {name: file_info(output_dir / name) for name in OUTPUTS[:3]}
    (output_dir / OUTPUTS[3]).write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def create_synthetic_inputs(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    examples = {
        'synthetic-low': ('The parties audit only records relating to this agreement; the sole consequence of audit failure is a written reminder; each party has identical audit rights and obligations.', .9),
        'synthetic-medium': ('The parties audit all books and records.', .9),
        'synthetic-high': ('The parties audit all books and records; audit costs are unlimited; one party alone may expand the audit scope; audit failure triggers termination of all services.', .9),
        'synthetic-no-detection': ('Routine project notes.', .01),
    }
    predictions, chunks = [], []
    for title, (text, probability) in examples.items():
        predictions.append({'title': title, 'chunk_id': 0,
                            **{category: probability if category == 'Audit Rights' else .01 for category in load_categories()}})
        chunks.append({'title': title, 'chunk_id': 0, 'chunk_text': text, 'char_start': 0, 'char_end': len(text)})
    predictions_path, chunks_path = directory / 'predictions.parquet', directory / 'chunks.parquet'
    pd.DataFrame(predictions).to_parquet(predictions_path, index=False)
    pd.DataFrame(chunks).to_parquet(chunks_path, index=False)
    return predictions_path, chunks_path


def main():
    parser = argparse.ArgumentParser(description='Run provisional contract risk scoring on non-test inputs.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--synthetic', action='store_true')
    mode.add_argument('--validation', action='store_true')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--chunks', type=Path)
    parser.add_argument('--split', choices=['development', 'validation'], default='development')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    audit = None
    if args.validation:
        if args.predictions or args.chunks:
            parser.error("--validation cannot be combined with input paths")
        run_dir = ROOT / "results/training/15ep/legal-bert-focal-optuna"
        sources = {
            "probabilities": run_dir / "val_probs.npy",
            "train_chunks": ROOT / "data/processed/train_chunks_labeled.parquet",
            "titles": ROOT / "data/processed/val_titles.json",
            "summary": run_dir / "summary.json",
            "metrics": run_dir / "val_per_category.csv",
        }
        prediction_rows, chunk_rows, audit = load_validation_inputs(*sources.values())
        audit["sources"] = {name: file_info(path) for name, path in sources.items()}
        audit["training_notebook"] = file_info(ROOT / "notebooks/pipeline/05_model_training.ipynb")
        input_dir = args.output_dir / "validation_inputs"
        input_dir.mkdir(parents=True, exist_ok=False)
        predictions, chunks = input_dir / "predictions.parquet", input_dir / "chunks.parquet"
        prediction_rows.to_parquet(predictions, index=False)
        chunk_rows.to_parquet(chunks, index=False)
        split = "validation"
    elif args.synthetic:
        if args.predictions or args.chunks:
            parser.error('--synthetic cannot be combined with input paths')
        predictions, chunks = create_synthetic_inputs(args.output_dir / 'synthetic_inputs')
        split = 'synthetic'
    else:
        if not args.predictions or not args.chunks:
            parser.error('Provide --predictions and --chunks, or use --synthetic')
        predictions, chunks, split = args.predictions, args.chunks, args.split
    manifest = run_pipeline(predictions, chunks, args.output_dir, args.threshold, split, alignment_audit=audit)
    print(json.dumps(manifest['counts'], indent=2))


if __name__ == '__main__':
    main()
