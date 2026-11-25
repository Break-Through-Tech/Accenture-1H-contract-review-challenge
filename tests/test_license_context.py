"""License relevance and use restrictions from validation, plus counterexamples."""
import unittest
from test_contextual_scoring import score
from test_evidence_quality import extract
from risk_scoring.scoring import score_clauses
from risk_scoring.aggregation import aggregate_contracts

DEFINITION = '2.4 "SOFTWARE" means the computer software, In machine executable object code format only, that is delivered and licensed by McDATA with the Product.'
DENIAL = 'c) Ownership: Agent acknowledges that Kallo is the sole owner of the Trademarks and nothing herein shall grant to Agent any right or interest in the Trademarks.'
MTI = "Reseller grants McDATA a license to use Reseller's trademarks and corporate logos solely for such marketing and reference purposes."
BLUEFLY = '12.1 License\n\nIBM grants Customer a nonexclusive, revocable license to use the Base Components solely in connection with the Services as provided under this Agreement.'
ACCURAY = 'Accuray and its licensors retain all intellectual property rights in the Products. Accuray hereby grants Distributor or Customer a nonexclusive, non-transferable, royalty-free right to use the software provided in connection with the Products only in machine readable form and only in combination with the Products with which such software is provided.'


class LicenseContextTests(unittest.TestCase):
    def test_definitions_and_denials_excluded(self):
        for text, reason in [(DEFINITION, 'license_definition_without_grant'), (DENIAL, 'license_grant_denied_not_established')]:
            result = score(text, 'License Grant')
            self.assertFalse(result['usable_for_scoring'])
            self.assertFalse(result['substantive_evidence'])
            self.assertIn(reason, result['review_reasons'])
            self.assertEqual(result['numeric_score_status'], 'deferred_review')
            self.assertIsNone(aggregate_contracts([result])[0]['triage_score'])

    def test_named_restrictions_change_only_scope(self):
        for text in [MTI, BLUEFLY, ACCURAY, MTI.replace(' ', '\n  ')]:
            result = score(text, 'License Grant')
            self.assertEqual(result['risk_scores'], dict(severity=1, scope=0, imbalance=0, exposure=1))
            self.assertTrue(result['usable_for_scoring'])
            for match in result['rule_matches']:
                self.assertEqual(text[match['char_start']:match['char_end']], match['quote'])

    def test_adjacent_grant_recovery_has_exact_nonzero_offsets(self):
        text = 'Intro. ' + DEFINITION + ' ' + MTI
        rows = extract(text, 'License Grant', split=len('Intro. ' + DEFINITION))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row['char_start'], row['char_end']), (7, len(text)))
        self.assertEqual(row['original_text'], text[7:])
        self.assertEqual(row['supporting_chunk_ids'], [0])
        self.assertTrue(score_clauses(rows)[0]['usable_for_scoring'])

    def test_recovery_does_not_jump_heading_or_unrelated_text(self):
        for between in ['Payment Terms.', 'Supplier shall deliver goods.']:
            text = DEFINITION + ' ' + between + ' ' + MTI
            rows = extract(text, 'License Grant', split=len(DEFINITION))
            self.assertEqual(rows[0]['original_text'], DEFINITION)
            self.assertFalse(score_clauses(rows)[0]['usable_for_scoring'])

    def test_recovered_exception_and_truncation_stay_deferred(self):
        for tail in [MTI + ' However, exceptions apply under Section 4.', MTI[:-1]]:
            text = DEFINITION + ' ' + tail
            rows = extract(text, 'License Grant', split=len(DEFINITION))
            self.assertFalse(score_clauses(rows)[0]['usable_for_scoring'])

    def test_negation_and_exceptions_never_reduce_scope(self):
        for text in ['Owner does not grant Customer a license to use Software solely in connection with the Services.',
                     'Nothing herein grants Customer a license to use Software solely in connection with the Services.',
                     'If approved, ' + MTI, MTI + ' Except as provided in Section 4.']:
            result = score(text, 'License Grant')
            self.assertFalse(result['usable_for_scoring'])
            self.assertEqual(result['risk_scores']['scope'], 1)

    def test_worldwide_and_restriction_defer_scope(self):
        result = score(MTI.replace('a license', 'a worldwide license'), 'License Grant')
        self.assertEqual(result['risk_scores']['scope'], 1)
        self.assertIn('license_scope_dimensions_conflict', result['review_reasons'])
        self.assertFalse(result['usable_for_scoring'])
        self.assertEqual(score('Owner grants Customer a worldwide license.', 'License Grant')['risk_scores']['scope'], 2)

    def test_unrelated_use_restrictions_not_applied_to_grant(self):
        for text in ['Owner grants Customer a license to use Software. Customer maintains records solely in connection with the Services.',
                     'Owner grants Customer a license to use Software and Customer shall pay fees solely in connection with the Services.']:
            self.assertEqual(score(text, 'License Grant')['risk_scores']['scope'], 1)

    def test_retained_ownership_and_grant_preserved(self):
        for text in [ACCURAY, DENIAL + ' ' + MTI,
                     'Owner retains ownership and grants Customer a license to use Software solely in connection with the Services.']:
            self.assertTrue(score(text, 'License Grant')['substantive_evidence'])
        self.assertTrue(score(DEFINITION, 'Warranty Duration')['usable_for_scoring'])
        self.assertTrue(score('Owner grants Customer a license to distribute Software by any means.', 'License Grant')['usable_for_scoring'])


if __name__ == '__main__':
    unittest.main()
