"""The three concrete cases found in manual validation review."""
import unittest

from test_evidence_quality import extract
from test_contextual_scoring import score
from risk_scoring.scoring import score_clauses
from risk_scoring.aggregation import aggregate_contracts

HEADING = '(b) Additional Action to Assign Interest.'
PROVISION = 'To the extent such work may not be deemed a "work for hire" under applicable law, the Developer hereby assigns to the Client all of its right, title, and interest in and to such work.'
LICENSE = '2TheMart hereby grants to i-Escrow a worldwide, non-exclusive right to use, reproduce, distribute, publicly perform, publicly display and digitally perform the 2TheMart Content soley with respect to  and in conjunction with the Co-Branded Site all with the prior written consent of 2TheMart, for the term of this Agreement.'
STATUS = 'The Distributor represents that the Distributor or a subsidiary of the Distributor is or will be fully licensed and registered in the Territory and will provide professional distribution services for the products of the Principal.'
GRANT = 'The Principal grants Distributor non- exclusive rights to sell these products within the countries of Romania (the "Territory"), which may be amended by mutual written agreement.'


class ManualReviewFixTests(unittest.TestCase):
    def test_assignment_heading_alone_excluded(self):
        rows = extract(HEADING, 'Ip Ownership Assignment')
        self.assertEqual(rows[0]['localization'], 'heading_only')
        self.assertFalse(rows[0]['usable_for_scoring'])
        self.assertIsNone(aggregate_contracts(score_clauses(rows))[0]['triage_score'])

    def test_assignment_heading_recovers_exact_provision(self):
        text = HEADING + ' ' + PROVISION
        rows = extract(text, 'Ip Ownership Assignment', split=len(HEADING))
        self.assertEqual(rows[0]['original_text'], text)
        self.assertEqual((rows[0]['char_start'], rows[0]['char_end']), (0, len(text)))
        result = score_clauses(rows)[0]
        self.assertFalse(result['usable_for_scoring'])
        self.assertTrue(result['needs_review'])

    def test_operative_assignment_is_preserved(self):
        text = 'The Developer hereby assigns all rights in the deliverables to the Client.'
        rows = extract(text, 'Ip Ownership Assignment')
        self.assertTrue(rows[0]['usable_for_scoring'])
        self.assertEqual(rows[0]['original_text'], text)

    def test_restricted_worldwide_exact_text(self):
        result = score(LICENSE, 'License Grant')
        self.assertEqual(result['original_text'], LICENSE)
        self.assertEqual(result['risk_scores']['scope'], 1)
        self.assertFalse(result['usable_for_scoring'])
        self.assertIn('wording_adjustment_requires_context', result['review_reasons'])
        for match in result['rule_matches']:
            self.assertEqual(LICENSE[match['char_start']:match['char_end']], match['quote'])

    def test_restrictions_individually_and_positive_grant(self):
        base = 'The owner grants a worldwide license'
        for restriction in [' soley for internal use.', ' with prior written consent.',
                            ' for the term of this Agreement.',
                            ' with respect to and in conjunction with the site.']:
            self.assertEqual(score(base + restriction, 'License Grant')['risk_scores']['scope'], 1)
        self.assertEqual(score(base + '.', 'License Grant')['risk_scores']['scope'], 2)
        self.assertEqual(score('The owner does not grant a worldwide license.', 'License Grant')['risk_scores']['scope'], 1)

    def test_status_representation_excluded(self):
        result = score_clauses(extract(STATUS, 'License Grant'))[0]
        self.assertEqual(result['original_text'], STATUS)
        self.assertFalse(result['substantive_evidence'])
        self.assertFalse(result['usable_for_scoring'])
        self.assertIn('licensing_status_is_not_a_rights_grant', result['review_reasons'])
        self.assertIsNone(aggregate_contracts([result])[0]['triage_score'])

    def test_actual_grant_in_same_passage_and_negated_counterexample(self):
        self.assertTrue(score(GRANT, 'License Grant')['usable_for_scoring'])
        self.assertTrue(score(STATUS + ' ' + GRANT, 'License Grant')['substantive_evidence'])
        result = score(STATUS + ' The Principal does not grant rights to distribute the products.', 'License Grant')
        self.assertFalse(result['usable_for_scoring'])


if __name__ == '__main__':
    unittest.main()
