"""Synthetic pipeline and output checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from risk_scoring.io import DEFAULT_PREDICTIONS, DEFAULT_CHUNKS
from risk_scoring.run import create_synthetic_inputs, run_pipeline, OUTPUTS, CLAUSE_COLUMNS


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
