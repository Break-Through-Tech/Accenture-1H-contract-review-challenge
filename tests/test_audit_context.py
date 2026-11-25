"""Audit wording contrasts; examples are not human risk labels."""
import unittest

from test_contextual_scoring import score

NARROW = 'Either party may inspect records relevant to invoices once annually, on ten business days written notice, at its own expense.'
BROAD = 'The supplier may inspect every financial record at any time, charge all inspection expenses to the customer, and immediately suspend services for refusal.'


class AuditContextTests(unittest.TestCase):
    def test_natural_narrow(self):
        result = score(NARROW, 'Audit Rights')
        self.assertEqual(result['risk_scores'], {'severity': 1, 'scope': 0, 'imbalance': 0, 'exposure': 1})
        self.assertEqual(result['signal_sources']['severity'], 'profile_default_unverified')
        self.assertEqual(result['signal_sources']['exposure'], 'profile_default_unverified')
        self.assertEqual(result['signal_sources']['imbalance'], 'wording_rule_provisional')

    def test_natural_broad(self):
        result = score(BROAD, 'Audit Rights')
        self.assertEqual(result['risk_scores'], {'severity': 2, 'scope': 2, 'imbalance': 1, 'exposure': 1})
        self.assertEqual(result['signal_sources']['exposure'], 'profile_default_unverified')
        self.assertTrue(result['usable_for_scoring'])

    def test_negation_exception_reference_and_question_defer(self):
        for text in ['The supplier may not inspect every financial record.',
                     BROAD + ' Except for the exclusions in Section 4.',
                     'Subject to Section 4, ' + BROAD,
                     'Could the supplier inspect every financial record?']:
            result = score(text, 'Audit Rights')
            self.assertFalse(result['usable_for_scoring'])
            self.assertFalse(any(rule['status'] == 'applied' for rule in result['rule_matches']))
            self.assertTrue(result['needs_review'])

    def test_broad_records_with_purpose_limit_deferred(self):
        result = score('The supplier may inspect all financial records relating to invoices only.', 'Audit Rights')
        self.assertEqual(result['risk_scores']['scope'], 1)
        self.assertFalse(result['usable_for_scoring'])

    def test_missing_reciprocity_is_not_proof_of_unilateral_permission(self):
        result = score('The supplier may inspect all financial records.', 'Audit Rights')
        self.assertEqual(result['signal_sources']['imbalance'], 'profile_default_unverified')
        exclusive = score('Only the customer may audit accounting records.', 'Audit Rights')
        self.assertEqual(exclusive['risk_scores']['imbalance'], 2)

    def test_cost_direction_does_not_favor_a_party(self):
        original = score(BROAD, 'Audit Rights')
        swapped = score(BROAD.replace('supplier', 'PLACEHOLDER').replace('customer', 'supplier').replace('PLACEHOLDER', 'customer'), 'Audit Rights')
        self.assertEqual(original['risk_scores'], swapped['risk_scores'])
        own_costs = score('The supplier may inspect records at its own expense.', 'Audit Rights')
        self.assertEqual(own_costs['signal_sources']['exposure'], 'profile_default_unverified')
        self.assertEqual(own_costs['signal_sources']['imbalance'], 'profile_default_unverified')

    def test_consequences_must_be_related_to_audit(self):
        result = score('The supplier may terminate services for late payment.', 'Audit Rights')
        self.assertEqual(result['signal_sources']['severity'], 'profile_default_unverified')
        termination = score(BROAD.replace('suspend', 'terminate'), 'Audit Rights')
        self.assertEqual(termination['risk_scores']['severity'], 2)

    def test_original_offsets_and_no_duplicate_points(self):
        result = score(BROAD, 'Audit Rights')
        for rule in result['rule_matches']:
            self.assertEqual(BROAD[rule['char_start']:rule['char_end']], rule['quote'])
        self.assertEqual(score(BROAD + ' ' + BROAD, 'Audit Rights')['risk_scores'], result['risk_scores'])

    def test_other_categories_unchanged(self):
        result = score(BROAD, 'License Grant')
        self.assertEqual(result['risk_scores'], result['starting_scores'])


if __name__ == '__main__':
    unittest.main()
