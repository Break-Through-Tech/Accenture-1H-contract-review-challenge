"""Validation excerpts used as development regressions, not human risk labels."""
import unittest

import pandas as pd

from risk_scoring.evidence import extract_evidence
from risk_scoring.io import load_categories, join_predictions
from risk_scoring.scoring import score_evidence

AUDIT = 'Once every twelve (12) months, 2TheMart through a CPA may inspect and audit such records to verify reports.'
LICENSE = 'i-Escrow hereby grants to 2TheMart a worldwide, non-exclusive right to use, reproduce, distribute, publicly perform, publicly display and digitally perform the i-Escrow Content on or in conjunction with 2TheMart auctions.'
CAP = 'EXCEPT FOR CLAIMS UNDER SECTION 9 HEREOF, THE LIABILITY OF A PARTY TO THE OTHER FOR DIRECT DAMAGES SHALL NOT EXCEED FIFTEEN MILLION DOLLARS.'


def score(text, category):
    return score_evidence({'category': category, 'original_text': text, 'char_start': 100,
                           'char_end': 100 + len(text), 'localization': 'heuristic_passage'})


class WordingRegressionTests(unittest.TestCase):
    def test_real_permission_with_frequency_limit(self):
        result = score(AUDIT, 'Audit Rights')
        self.assertEqual(result['risk_scores']['scope'], 1)
        self.assertTrue(any(rule['status'] == 'observed_context_only' for rule in result['rule_matches']))

    def test_real_worldwide_license_and_original_offsets(self):
        for text in [LICENSE, LICENSE.replace(' ', '\n  ')]:
            result = score(text, 'License Grant')
            self.assertEqual(result['risk_scores']['scope'], 2)
            for rule in result['rule_matches']:
                self.assertEqual(text[rule['char_start']-100:rule['char_end']-100], rule['quote'])

    def test_real_cap_with_exception_deferred(self):
        result = score(CAP, 'Uncapped Liability')
        self.assertEqual(result['risk_scores']['exposure'], 2)
        self.assertTrue(result['rule_matches'])
        self.assertTrue(all(rule['status'] == 'deferred_uncertainty' for rule in result['rule_matches']))

    def test_negated_license_and_hypothetical_deferred(self):
        for prefix in ['The party does not ', 'If approved, the party ', 'The party could ']:
            result = score(prefix + 'grant a worldwide license.', 'License Grant')
            self.assertEqual(result['risk_scores']['scope'], 1)

    def test_unrelated_negation_does_not_block_supported_sentence(self):
        result = score('No implied licenses are granted. ' + LICENSE, 'License Grant')
        self.assertEqual(result['risk_scores']['scope'], 2)
        self.assertTrue(result['needs_review'])

    def test_adjacent_qualifications_still_block(self):
        for text in ['Except as agreed in Section 4. ' + LICENSE,
                     LICENSE + ' However, this grant excludes overseas use.']:
            self.assertEqual(score(text, 'License Grant')['risk_scores']['scope'], 1)

    def test_cap_negative_construction_is_not_negated_cap(self):
        text = 'Aggregate liability shall not exceed the total fees paid.'
        self.assertEqual(score(text, 'Uncapped Liability')['risk_scores']['exposure'], 1)
        self.assertEqual(score('It is not true that ' + text, 'Uncapped Liability')['risk_scores']['exposure'], 2)

    def test_heading_retains_next_provision(self):
        text = '5.3 AUDIT RIGHTS. ' + AUDIT
        categories = load_categories()
        predictions = pd.DataFrame([{'title': 'fixture', 'chunk_id': 0,
                                     **{category: .9 if category == 'Audit Rights' else .01 for category in categories}}])
        chunks = pd.DataFrame([{'title': 'fixture', 'chunk_id': 0, 'chunk_text': text,
                                'char_start': 0, 'char_end': len(text)}])
        evidence = extract_evidence(join_predictions(predictions, chunks, categories), categories)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]['original_text'], text)
        self.assertEqual(score_evidence(evidence[0])['risk_scores']['scope'], 1)

    def test_supported_consequence_and_discretion(self):
        result = score('Any assignment without prior consent shall be null and void.', 'Anti-Assignment')
        self.assertEqual(result['risk_scores']['severity'], 1)
        result = score('Consent may be withheld in its sole discretion.', 'Anti-Assignment')
        self.assertEqual(result['risk_scores']['imbalance'], 1)


if __name__ == '__main__':
    unittest.main()
