"""Synthetic pipeline and output checks."""
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from risk_scoring.io import DEFAULT_PREDICTIONS, DEFAULT_CHUNKS
from risk_scoring.run import create_synthetic_inputs, run_pipeline, OUTPUTS, CLAUSE_COLUMNS, assessment_fields
from test_contextual_scoring import score


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.predictions, self.chunks = create_synthetic_inputs(self.root / 'inputs')

    def test_outputs_and_traceability(self):
        output = self.root / 'run'
        manifest = run_pipeline(self.predictions, self.chunks, output, data_split='synthetic')
        self.assertEqual({path.name for path in output.iterdir()}, set(OUTPUTS))
        self.assertEqual(manifest['counts']['contract_risk_distribution'], {'Low': 1, 'Medium': 1, 'High': 1, 'Review Required': 1})
        clauses = pd.read_parquet(output / 'clause_risks.parquet')
        contracts = pd.read_csv(output / 'contract_risks.csv')
        flags = pd.read_csv(output / 'coverage_flags.csv')
        self.assertEqual(len(clauses), 3)
        self.assertEqual(len(contracts), 4)
        self.assertEqual(list(clauses.columns), CLAUSE_COLUMNS)
        for row in clauses.to_dict('records'):
            original = json.loads(row['finding_json'])
            self.assertEqual(original['rule_matches'][0]['status'], 'applied')
            self.assertEqual(original['risk_total'], row['risk_total'])
            self.assertEqual(original['max_detection_probability'], .9)
            self.assertTrue(row['evidence_group_id'])
        missing = flags[flags.flag == 'not_detected_at_threshold']
        self.assertEqual(len(missing), 161)
        self.assertTrue(missing.detail.str.contains('Not proof of absence').all())
        self.assertEqual(json.loads((output / 'run_manifest.json').read_text()), manifest)
        self.assertEqual(len(manifest['configuration']['detection_thresholds']), 41)
        self.assertTrue(all(len(info['sha256']) == 64 for info in manifest['inputs'].values()))

    def test_default_zero_is_not_verified_and_scoring_is_untouched(self):
        finding = score('The license covers the project.', 'License Grant')
        original = deepcopy(finding)
        fields = assessment_fields(finding)
        self.assertEqual(fields['imbalance_assessment_status'], 'unverified_default')
        self.assertEqual(fields['assessment_status'], 'human_review_required')
        self.assertEqual(fields['score_basis'], 'provisional_triage')
        self.assertEqual(finding, original)

    def test_one_supported_signal_does_not_establish_full_assessment(self):
        finding = score('Either party may terminate this Agreement for convenience.', 'Termination For Convenience')
        fields = assessment_fields(finding)
        self.assertEqual(fields['imbalance_assessment_status'], 'wording_rule_supported')
        self.assertEqual(fields['severity_assessment_status'], 'unverified_default')
        self.assertEqual(fields['assessment_status'], 'human_review_required')

    def test_uncertainty_and_conflicts_override_support(self):
        for text in ['Audit costs are unlimited except as provided in Section 4.',
                     'Audit costs are unlimited. Audit costs are limited to a fixed fee.']:
            fields = assessment_fields(score(text, 'Audit Rights'))
            self.assertEqual(fields['exposure_assessment_status'], 'deferred_for_review')
            self.assertEqual(fields['assessment_status'], 'human_review_required')
        finding = score('The parties audit all books and records.', 'Audit Rights')
        finding['numeric_score_status'] = 'deferred_review'
        self.assertEqual(assessment_fields(finding)['scope_assessment_status'], 'deferred_for_review')

    def test_assessment_export_does_not_replace_triage(self):
        output = self.root / 'assessments'
        run_pipeline(self.predictions, self.chunks, output, data_split='synthetic')
        clauses = pd.read_parquet(output / 'clause_risks.parquet')
        contracts = pd.read_csv(output / 'contract_risks.csv').set_index('contract_id')
        self.assertTrue(clauses.score_basis.eq('provisional_triage').all())
        self.assertTrue(contracts.score_basis.eq('provisional_triage').all())
        self.assertEqual(contracts.loc['synthetic-medium', 'risk_level'], 'Medium')
        self.assertEqual(contracts.loc['synthetic-medium', 'assessment_status'], 'human_review_required')
        self.assertEqual(contracts.loc['synthetic-no-detection', 'assessment_status'], 'human_review_required')
        self.assertEqual(contracts.loc['synthetic-high', 'assessment_status'], 'wording_supported_not_human_reviewed')
        for row in clauses.to_dict('records'):
            self.assertEqual(row['risk_total'], json.loads(row['finding_json'])['risk_total'])
            self.assertEqual(row['assessment_status'], assessment_fields(json.loads(row['finding_json']))['assessment_status'])

    def test_empty_detections_have_stable_outputs(self):
        output = self.root / 'empty'
        manifest = run_pipeline(self.predictions, self.chunks, output, threshold=1, data_split='synthetic')
        clauses = pd.read_parquet(output / 'clause_risks.parquet')
        self.assertEqual(list(clauses.columns), CLAUSE_COLUMNS)
        self.assertEqual(len(clauses), 0)
        self.assertEqual(manifest['counts']['contract_risk_distribution']['Review Required'], 4)
        self.assertTrue(pd.read_csv(output / 'contract_risks.csv').triage_score.isna().all())

    def test_invalid_columns_and_threshold_fail_before_outputs(self):
        output = self.root / 'bad'
        with self.assertRaises(ValueError):
            run_pipeline(self.predictions, self.chunks, output, threshold=float('nan'))
        frame = pd.read_parquet(self.predictions).drop(columns='Audit Rights')
        frame.to_parquet(self.predictions, index=False)
        with self.assertRaises(ValueError):
            run_pipeline(self.predictions, self.chunks, output)
        self.assertFalse(output.exists())

    def test_test_inputs_blocked_without_loading(self):
        with self.assertRaises(ValueError):
            run_pipeline(DEFAULT_PREDICTIONS, DEFAULT_CHUNKS, self.root / 'blocked')
        self.assertFalse((self.root / 'blocked').exists())

    def test_reproducible_data_and_no_overwrite(self):
        first, second = self.root / 'first', self.root / 'second'
        run_pipeline(self.predictions, self.chunks, first, data_split='synthetic')
        run_pipeline(self.predictions, self.chunks, second, data_split='synthetic')
        for name in OUTPUTS[:3]:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        with self.assertRaises(FileExistsError):
            run_pipeline(self.predictions, self.chunks, first)

    def test_command_line(self):
        result = subprocess.run([sys.executable, '-B', '-m', 'risk_scoring.run', '--synthetic',
                                 '--output-dir', str(self.root / 'cli')], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)['contracts'], 4)


if __name__ == '__main__':
    unittest.main()
