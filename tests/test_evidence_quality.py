"""Evidence eligibility and concrete review reasons."""
import unittest

import pandas as pd

from risk_scoring.io import load_categories, join_predictions
from risk_scoring.evidence import extract_evidence
from risk_scoring.scoring import score_clauses
from risk_scoring.aggregation import aggregate_contracts


def extract(text, category, split=None):
    categories = load_categories()
    windows = [(0, len(text))] if split is None else [(0, split), (split, len(text))]
    predictions, chunks = [], []
    for chunk_id, (start, end) in enumerate(windows):
        predictions.append({'title': 'example', 'chunk_id': chunk_id,
                            **{name: .9 if name == category and chunk_id == 0 else .01 for name in categories}})
        chunks.append({'title': 'example', 'chunk_id': chunk_id, 'char_start': start,
                       'char_end': end, 'chunk_text': text[start:end]})
    return extract_evidence(join_predictions(pd.DataFrame(predictions), pd.DataFrame(chunks), categories), categories)


class EvidenceQualityTests(unittest.TestCase):
    def assert_excluded(self, rows):
        scored = score_clauses(rows)
        self.assertTrue(all(row['needs_review'] for row in scored))
        self.assertTrue(all(not row['usable_for_scoring'] for row in scored))
        summary = aggregate_contracts(scored)[0]
        self.assertIsNone(summary['triage_score'])
        self.assertEqual(summary['risk_level'], 'Review Required')
        self.assertEqual(summary['evidence_group_count'], 0)

    def test_heading_only_excluded(self):
        rows = extract('Intellectual Property, Inventions and Patents.', 'Ip Ownership Assignment')
        self.assertEqual(rows[0]['localization'], 'heading_only')
        self.assert_excluded(rows)

    def test_contents_excluded_without_following_unrelated_text(self):
        rows = extract('Audit Rights ........ 16\n\nThe supplier shall deliver goods.', 'Audit Rights')
        self.assertEqual(rows[0]['localization'], 'table_of_contents')
        self.assert_excluded(rows)

    def test_misleading_escrow_name_and_source_code_alone(self):
        for text in ['The buyer pays i-Escrow the agreed upon price.', 'The source code is owned by the supplier.']:
            self.assert_excluded(extract(text, 'Source Code Escrow'))

    def test_source_code_escrow_positive(self):
        rows = extract('The supplier shall deposit source code with the escrow agent.', 'Source Code Escrow')
        self.assertTrue(rows[0]['usable_for_scoring'])

    def test_real_heading_recovers_following_provision_across_chunk(self):
        heading = 'Intellectual Property, Inventions and Patents.'
        text = heading + ' CONSULTANT acknowledges that all inventions belong to the Company.'
        rows = extract(text, 'Ip Ownership Assignment', split=len(heading))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['original_text'], text)
        self.assertEqual(rows[0]['char_end'], len(text))
        self.assertEqual(rows[0]['supporting_chunk_ids'], [0])
        self.assertTrue(rows[0]['usable_for_scoring'])

    def test_clean_evidence_is_provisional_without_review_issue(self):
        scored = score_clauses(extract('The parties audit all books and records.', 'Audit Rights'))
        self.assertFalse(scored[0]['needs_review'])
        self.assertEqual(scored[0]['review_reasons'], [])
        self.assertIn('provisional_rubric_and_wording_rules', scored[0]['limitations'])
        summary = aggregate_contracts(scored)[0]
        self.assertFalse(summary['needs_review'])
        self.assertEqual(summary['risk_status'], 'provisional_heuristic_not_human_reviewed')

    def test_exception_still_needs_review(self):
        scored = score_clauses(extract('Audit costs are unlimited except as provided in Section 4.', 'Audit Rights'))
        self.assertTrue(scored[0]['needs_review'])
        self.assertIn('negation_condition_or_reference_unresolved', scored[0]['review_reasons'])

    def test_explicit_exclusion_cannot_raise_contract_score(self):
        scored = score_clauses(extract('Audit costs are unlimited.', 'Audit Rights'))
        scored[0]['usable_for_scoring'] = False
        self.assertEqual(aggregate_contracts(scored)[0]['risk_level'], 'Review Required')


if __name__ == '__main__':
    unittest.main()
