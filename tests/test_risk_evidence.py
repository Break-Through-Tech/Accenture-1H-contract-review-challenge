"""Synthetic fixtures test implementation behavior, not human risk judgments."""
import unittest
import tempfile
from pathlib import Path

import pandas as pd

from risk_scoring.io import join_predictions, load_categories, load_prediction_chunks
from risk_scoring.evidence import extract_evidence


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.categories = load_categories()

    def fixture(self, text='Audit rights apply. Other terms apply.', windows=None, category='Audit Rights'):
        windows = windows or [(0, len(text))]
        chunks, predictions = [], []
        for chunk_id, (start, end) in enumerate(windows):
            chunks.append(dict(title='contract-A', chunk_id=chunk_id,
                               char_start=start, char_end=end, chunk_text=text[start:end],
                               **{c: 1 for c in self.categories}))
            predictions.append(dict(title='contract-A', chunk_id=chunk_id,
                                    **{c: .8 if c == category else .01 for c in self.categories}))
        return pd.DataFrame(predictions), pd.DataFrame(chunks)

    def joined(self, **kwargs):
        p, c = self.fixture(**kwargs)
        return join_predictions(p, c, self.categories)

    def test_labels_excluded_and_join_by_key(self):
        p, c = self.fixture(windows=[(0, 19), (19, 37)])
        result = join_predictions(p.iloc[::-1], c, self.categories)
        self.assertEqual(result['Audit Rights'].tolist(), [.8, .8])
        self.assertEqual(result.chunk_id.tolist(), [0, 1])
        self.assertFalse(any(col.endswith('_x') for col in result))

    def test_schema_keys_probabilities_offsets(self):
        p, c = self.fixture()
        for bad in [p.drop(columns=self.categories[0]), p.assign(extra=0),
                    pd.concat([p, p]), p.assign(**{'Audit Rights': 1.1}),
                    p.assign(**{'Audit Rights': float('nan')})]:
            with self.assertRaises(ValueError):
                join_predictions(bad, c, self.categories)
        for bad in [c.assign(title='other'), c.assign(char_end=999), pd.concat([c, c])]:
            with self.assertRaises(ValueError):
                join_predictions(p, bad, self.categories)

    def test_localization_and_original_offsets(self):
        text = 'Intro. Audit rights apply. Other terms apply.'
        records = extract_evidence(self.joined(text=text), self.categories)
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r['original_text'], 'Audit rights apply.')
        self.assertEqual(text[r['char_start']:r['char_end']], r['original_text'])
        self.assertFalse(r['needs_review'])
        self.assertIn('heuristic_localization_not_verified', r['limitations'])
        self.assertEqual(r['threshold_status'], 'provisional')

    def test_duplicate_support_and_order_invariance(self):
        text = 'Intro. Audit rights apply. Other terms apply.'
        frame = self.joined(text=text, windows=[(0, len(text)), (7, len(text))])
        records = extract_evidence(frame, self.categories)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['supporting_chunk_ids'], [0, 1])
        self.assertEqual(records[0]['max_detection_probability'], .8)
        self.assertEqual(records, extract_evidence(frame.iloc[::-1], self.categories))

    def test_separate_occurrences_not_merged(self):
        text = 'Audit rights apply. Other terms apply. Audit rights apply.'
        self.assertEqual(len(extract_evidence(self.joined(text=text), self.categories)), 2)

    def test_fallback_and_thresholds(self):
        frame = self.joined(text='No relevant cue appears here.')
        records = extract_evidence(frame, self.categories)
        self.assertEqual(records[0]['localization'], 'chunk_fallback')
        self.assertIn('no_localization_cue', records[0]['review_reasons'])
        self.assertEqual(extract_evidence(frame, self.categories, threshold=.9), [])
        self.assertEqual(len(extract_evidence(frame, self.categories, threshold=.8)), 1)
        for threshold in [-1, float('nan'), {'Audit Rights': .5}]:
            with self.assertRaises(ValueError):
                extract_evidence(frame, self.categories, threshold=threshold)

    def test_qualification_retained_without_interpreting_negation(self):
        text = 'Intro. No audit rights are granted. Except as provided in Section 4. End.'
        r = extract_evidence(self.joined(text=text), self.categories)[0]
        self.assertIn('No audit rights', r['original_text'])
        self.assertIn('Except as provided', r['original_text'])
        self.assertIn('qualification_or_reference_requires_review', r['review_reasons'])
        self.assertNotIn('risk_score', r)

    def test_conflicting_source_overlap_rejected(self):
        p, c = self.fixture(windows=[(0, 19), (0, 19)])
        c.loc[1, 'chunk_text'] = 'X' + c.loc[1, 'chunk_text'][1:]
        with self.assertRaises(ValueError):
            extract_evidence(join_predictions(p, c, self.categories), self.categories)

    def test_contract_and_category_isolation(self):
        frame = self.joined()
        frame['License Grant'] = .8
        other = frame.copy()
        other['title'] = 'contract-B'
        records = extract_evidence(pd.concat([frame, other]), self.categories)
        self.assertEqual(len(records), 4)
        self.assertEqual(len({r['evidence_id'] for r in records}), 4)

    def test_paragraph_boundary(self):
        text = 'Introduction\n\nAudit rights apply\n\nOther terms'
        r = extract_evidence(self.joined(text=text), self.categories)[0]
        self.assertEqual(r['original_text'], 'Audit rights apply')

    def test_file_input_structure_only(self):
        # Exercise the real Parquet loader with reproducible, label-bearing inputs.
        predictions, chunks = self.fixture()
        predictions = pd.concat([predictions, predictions.assign(title='contract-B')], ignore_index=True)
        chunks = pd.concat([chunks, chunks.assign(title='contract-B')], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory)
            predictions.iloc[::-1].to_parquet(inputs / 'predictions.parquet', index=False)
            chunks.to_parquet(inputs / 'chunks.parquet', index=False)
            frame, categories = load_prediction_chunks(inputs / 'predictions.parquet', inputs / 'chunks.parquet')
        self.assertEqual(frame.shape, (2, 46))
        self.assertEqual(frame.title.nunique(), 2)
        self.assertEqual(categories, self.categories)
        self.assertEqual(frame.title.tolist(), ['contract-A', 'contract-B'])
        self.assertEqual(frame['Audit Rights'].tolist(), [.8, .8])


if __name__ == '__main__':
    unittest.main()
