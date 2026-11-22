"""Audit saved validation alignment without changing model or risk rules."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from risk_scoring.io import ROOT, load_validation_inputs, TEXT_COLUMNS


class ValidationAlignmentTests(unittest.TestCase):
    def setUp(self):
        run = ROOT / 'results/training/15ep/legal-bert-focal-optuna'
        self.paths = [run / 'val_probs.npy', ROOT / 'data/processed/train_chunks_labeled.parquet',
                      ROOT / 'data/processed/val_titles.json', run / 'summary.json', run / 'val_per_category.csv']
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.temp = Path(self.directory.name)

    def test_saved_order_and_label_free_export(self):
        predictions, chunks, audit = load_validation_inputs(*self.paths)
        self.assertEqual(predictions.shape, (2069, 43))
        self.assertEqual(list(chunks.columns), TEXT_COLUMNS)
        self.assertEqual(audit['contracts'], 61)
        self.assertLess(audit['max_ap_difference'], 1e-10)
        pd.testing.assert_frame_equal(predictions[['title', 'chunk_id']], chunks[['title', 'chunk_id']])
        train = pd.read_parquet(self.paths[1])
        titles = json.loads(self.paths[2].read_text())['val_titles']
        expected = train[train.title.isin(titles)][['title', 'chunk_id']].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, predictions[['title', 'chunk_id']])

    def test_same_shape_reversed_rows_rejected(self):
        path = self.temp / 'reversed.npy'
        np.save(path, np.load(self.paths[0])[::-1])
        with self.assertRaisesRegex(ValueError, 'AP'):
            load_validation_inputs(path, *self.paths[1:])

    def test_same_shape_reversed_columns_rejected(self):
        path = self.temp / 'reversed.npy'
        np.save(path, np.load(self.paths[0])[:, ::-1])
        with self.assertRaisesRegex(ValueError, 'AP'):
            load_validation_inputs(path, *self.paths[1:])

    def test_subset_run_rejected(self):
        summary = json.loads(self.paths[3].read_text())
        summary['config']['subset'] = {'validation': 2069}
        path = self.temp / 'summary.json'
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'subset'):
            load_validation_inputs(*self.paths[:3], path, self.paths[4])

    def test_category_metadata_order_rejected(self):
        path = self.temp / 'metrics.csv'
        pd.read_csv(self.paths[4], index_col=0).iloc[::-1].to_csv(path)
        with self.assertRaisesRegex(ValueError, 'category order'):
            load_validation_inputs(*self.paths[:4], path)


if __name__ == '__main__':
    unittest.main()
