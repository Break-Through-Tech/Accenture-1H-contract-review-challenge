"""Reproductions of the six read-only audit findings."""
import tempfile
from pathlib import Path
import unittest

import pandas as pd

from risk_scoring.scoring import score_clauses
from risk_scoring.aggregation import aggregate_contracts
from risk_scoring.run import create_synthetic_inputs, run_pipeline
from test_evidence_quality import extract
from test_contextual_scoring import score


class AuditFixTests(unittest.TestCase):
    def test_heading_recovery_retains_cross_chunk_exception(self):
        text = 'AUDIT RIGHTS. Audit costs are unlimited. Except that audit costs are capped at $100.'
        records = extract(text, 'Audit Rights', split=len('AUDIT RIGHTS.'))
        self.assertEqual(records[0]['original_text'], text)
        result = score_clauses(records)[0]
        self.assertEqual(result['risk_scores']['exposure'], 1)
        self.assertFalse(result['usable_for_scoring'])
        self.assertTrue(result['needs_review'])
        self.assertIsNone(aggregate_contracts([result])[0]['triage_score'])

    def test_short_qualification_is_retained(self):
        text = 'AUDIT RIGHTS. Audit costs are unlimited. Except for routine audits.'
        records = extract(text, 'Audit Rights', split=len('AUDIT RIGHTS.'))
        self.assertEqual(records[0]['original_text'], text)
        self.assertFalse(score_clauses(records)[0]['usable_for_scoring'])

    def test_heading_recovery_stops_at_new_section(self):
        text = 'AUDIT RIGHTS. INSURANCE. The supplier shall maintain insurance.'
        records = extract(text, 'Audit Rights')
        self.assertEqual(records[0]['original_text'], 'AUDIT RIGHTS.')
        self.assertEqual(records[0]['localization'], 'heading_only')
        self.assertFalse(records[0]['usable_for_scoring'])
        self.assertEqual(aggregate_contracts(score_clauses(records))[0]['risk_level'], 'Review Required')

    def test_incomplete_recovery_is_excluded(self):
        text = 'AUDIT RIGHTS. Audit costs are unlimited unless'
        records = extract(text, 'Audit Rights', split=len('AUDIT RIGHTS.'))
        self.assertFalse(records[0]['usable_for_scoring'])
        self.assertEqual(aggregate_contracts(score_clauses(records))[0]['risk_level'], 'Review Required')

    def test_negated_insurance_is_not_positive_evidence(self):
        result = score('Neither party shall maintain insurance.', 'Insurance')
        self.assertFalse(result['usable_for_scoring'])
        self.assertEqual(aggregate_contracts([result])[0]['risk_level'], 'Review Required')
        positive = score('Each party shall maintain insurance.', 'Insurance')
        self.assertTrue(positive['usable_for_scoring'])

    def test_abbreviation_does_not_hide_negation(self):
        text = 'It is not true that Acme Inc. grants a worldwide license.'
        result = score(text, 'License Grant')
        self.assertEqual(result['risk_scores']['scope'], 1)
        self.assertFalse(result['usable_for_scoring'])
        extracted = score_clauses(extract(text, 'License Grant'))
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0]['original_text'], text)
        self.assertFalse(extracted[0]['usable_for_scoring'])
        positive = score('Acme Inc. grants a worldwide license.', 'License Grant')
        self.assertEqual(positive['risk_scores']['scope'], 2)
        self.assertTrue(positive['usable_for_scoring'])

    def test_discussion_question_and_hypothesis_are_not_obligations(self):
        for text in ['The parties discussed whether liability is unlimited.',
                     'Liability is unlimited?', 'Assuming liability is unlimited, the parties will negotiate.']:
            result = score(text, 'Uncapped Liability')
            self.assertFalse(result['usable_for_scoring'])
            self.assertTrue(result['needs_review'])
            self.assertFalse(any(rule['status'] == 'applied' for rule in result['rule_matches']))
        self.assertTrue(score('Liability is unlimited.', 'Uncapped Liability')['usable_for_scoring'])

    def test_export_identifies_primary_contributors_and_provisional_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, chunks = create_synthetic_inputs(root / 'inputs')
            p, c = pd.read_parquet(predictions), pd.read_parquet(chunks)
            p.loc[p.title == 'synthetic-no-detection', 'Agreement Date'] = .9
            for title, text in [('synthetic-no-detection', 'This agreement is dated today.'),
                                ('synthetic-medium', 'The parties audit all books and records. Customer may inspect Supplier records.')]:
                c.loc[c.title == title, 'chunk_text'] = text
                c.loc[c.title == title, 'char_end'] = len(text)
            p.to_parquet(predictions, index=False)
            c.to_parquet(chunks, index=False)
            run_pipeline(predictions, chunks, root / 'outputs', data_split='synthetic')
            rows = pd.read_parquet(root / 'outputs/clause_risks.parquet')
            info = rows[rows.category == 'Agreement Date'].iloc[0]
            self.assertTrue(info.included_in_risk_groups)
            self.assertFalse(info.eligible_for_primary_triage)
            self.assertFalse(info.included_in_triage)
            medium = rows[rows.contract_id == 'synthetic-medium']
            self.assertEqual(medium.included_in_triage.sum(), 1)
            self.assertEqual(medium.eligible_for_primary_triage.sum(), 2)
            self.assertTrue(rows.risk_status.eq('provisional_heuristic_not_human_reviewed').all())
            self.assertIn('numeric_score_status', rows)


if __name__ == '__main__':
    unittest.main()
