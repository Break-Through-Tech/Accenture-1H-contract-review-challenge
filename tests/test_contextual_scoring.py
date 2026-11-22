"""Contextual scoring and unsupported-default regressions."""
import unittest

from risk_scoring.scoring import score_evidence
from risk_scoring.aggregation import aggregate_contracts


def score(text, category):
    return score_evidence({'contract_id': 'example', 'title': 'example', 'category': category,
                           'original_text': text, 'char_start': 0, 'char_end': len(text),
                           'localization': 'heuristic_passage', 'usable_for_scoring': True})


class ContextualScoringTests(unittest.TestCase):
    def test_informational_only_contract_requires_review(self):
        rows = [score('This Agreement is effective today.', 'Effective Date')]
        result = aggregate_contracts(rows)[0]
        self.assertIsNone(result['triage_score'])
        self.assertEqual(result['risk_level'], 'Review Required')
        self.assertEqual(result['evidence_group_counts']['Low'], 1)

    def test_worldwide_with_internal_use_is_not_unrestricted(self):
        result = score('IBM grants Customer a worldwide license within Customer enterprise only.', 'License Grant')
        self.assertEqual(result['risk_scores']['scope'], 1)
        self.assertEqual(result['signal_sources']['scope'], 'profile_default_unverified')
        self.assertIn('wording_adjustment_requires_context', result['review_reasons'])
        positive = score('IBM grants Customer a worldwide license.', 'License Grant')
        self.assertEqual(positive['risk_scores']['scope'], 2)

    def test_unlimited_requires_explicit_unqualified_evidence(self):
        for text in ['Neither party is liable for consequential damages.',
                     'Liability is unlimited except for the exclusions in Section 4.',
                     'Liability is not unlimited.']:
            result = score(text, 'Uncapped Liability')
            self.assertFalse(result['usable_for_scoring'])
            self.assertEqual(aggregate_contracts([result])[0]['risk_level'], 'Review Required')
        result = score('Liability is unlimited.', 'Uncapped Liability')
        self.assertTrue(result['usable_for_scoring'])
        self.assertEqual(result['signal_sources']['exposure'], 'wording_rule_provisional')
        self.assertEqual(result['signal_sources']['severity'], 'profile_default_unverified')

    def test_incidental_insurance_is_excluded(self):
        result = score('If no such request is made, Accuray shall ship in accordance with any instructions contained in the Purchase Order or via FedEx ground, with no extra insurance.', 'Insurance')
        self.assertFalse(result['usable_for_scoring'])
        self.assertEqual(aggregate_contracts([result])[0]['risk_level'], 'Review Required')
        positive = score('Each party agrees to carry liability insurance sufficient to cover the risks.', 'Insurance')
        self.assertTrue(positive['usable_for_scoring'])
        self.assertEqual(positive['signal_sources']['imbalance'], 'wording_rule_provisional')
        self.assertIn('financial_exposure_unverified', positive['review_reasons'])

    def test_unknown_balance_remains_unknown(self):
        result = score('The license covers the project.', 'License Grant')
        self.assertEqual(result['risk_scores']['imbalance'], 0)
        self.assertEqual(result['signal_sources']['imbalance'], 'profile_default_unverified')
        self.assertIn('party_balance_unknown_not_confirmed', result['limitations'])

    def test_ip_scope_positive_and_negated(self):
        narrow = score('The assignment covers solely the listed deliverables.', 'Ip Ownership Assignment')
        broad = score('The assignment includes all background and future inventions.', 'Ip Ownership Assignment')
        self.assertEqual(narrow['risk_scores']['scope'], 0)
        self.assertEqual(broad['risk_scores']['scope'], 2)
        self.assertEqual(broad['risk_scores']['severity'], 1)
        negated = score('The assignment does not include all background and future inventions.', 'Ip Ownership Assignment')
        self.assertEqual(negated['risk_scores']['scope'], 1)

    def test_insurance_scope_and_counterexample(self):
        result = score('The supplier shall maintain insurance that covers only project operations.', 'Insurance')
        self.assertEqual(result['risk_scores']['scope'], 0)
        self.assertTrue(result['usable_for_scoring'])
        negated = score('The supplier shall maintain insurance that is not limited to project operations.', 'Insurance')
        self.assertEqual(negated['risk_scores']['scope'], 1)

    def test_minimum_commitment_distinguishes_scope_from_exposure(self):
        result = score('The minimum purchase covers all products. The minimum commitment is capped at 10000 dollars.', 'Minimum Commitment')
        self.assertEqual(result['risk_scores']['scope'], 2)
        self.assertEqual(result['signal_sources']['exposure'], 'wording_rule_provisional')
        self.assertIn('financial_materiality_unknown', result['review_reasons'])
        negated = score('The minimum purchase does not cover all products.', 'Minimum Commitment')
        self.assertEqual(negated['risk_scores']['scope'], 1)

    def test_sharing_scope_and_explicit_discretion(self):
        narrow = score('Royalties apply solely to the licensed product.', 'Revenue/Profit Sharing')
        unilateral = score('One party alone may change the royalty rate.', 'Revenue/Profit Sharing')
        self.assertEqual(narrow['risk_scores']['scope'], 0)
        self.assertEqual(unilateral['risk_scores']['imbalance'], 2)
        self.assertEqual(unilateral['risk_scores']['exposure'], 1)
        negated = score('No party alone may change the royalty rate.', 'Revenue/Profit Sharing')
        self.assertEqual(negated['risk_scores']['imbalance'], 0)


if __name__ == '__main__':
    unittest.main()
