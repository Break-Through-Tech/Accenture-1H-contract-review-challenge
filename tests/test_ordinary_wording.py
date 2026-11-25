"""Validation wording regressions, not independently labeled risk examples."""
import unittest

from test_contextual_scoring import score
from test_evidence_quality import extract
from risk_scoring.scoring import score_evidence, score_clauses


NETGEAR = 'Solely for this purpose, NETGEAR and Bay Networks grant Distributor a non-exclusive, royalty-free, limited right to use the Trademarks.'
MTI = '4.1 TERMINATION WITHOUT CAUSE. Either party may terminate this Agreement without cause upon sixty (60) days prior written notice to the other party.'


class OrdinaryWordingTests(unittest.TestCase):
    def test_netgear_only_scope_changes(self):
        result = score_clauses(extract(NETGEAR, 'License Grant'))[0]
        self.assertEqual(result['risk_scores'], dict(severity=1, scope=0, imbalance=0, exposure=1))
        self.assertEqual(result['risk_total'], 2)
        self.assertTrue(result['usable_for_scoring'])
        self.assertEqual(result['signal_sources']['scope'], 'wording_rule_provisional')
        self.assertEqual(result['signal_sources']['exposure'], 'profile_default_unverified')

    def test_mti_supports_balance_without_lowering_total(self):
        result = score_clauses(extract(MTI, 'Termination For Convenience'))[0]
        self.assertEqual(result['risk_scores'], dict(severity=1, scope=1, imbalance=0, exposure=1))
        self.assertEqual((result['risk_total'], result['risk_level']), (3, 'Medium'))
        self.assertTrue(result['usable_for_scoring'])
        self.assertEqual(result['signal_sources']['imbalance'], 'wording_rule_provisional')

    def test_purpose_grant_variants_and_offsets(self):
        for text in [NETGEAR, NETGEAR.replace(' ', '\n  '),
                     'Solely for this purpose, Owner grants Customer a license to use Software.']:
            result = score_evidence(dict(category='License Grant', original_text=text,
                                        char_start=37, char_end=37+len(text), localization='heuristic_passage'))
            self.assertEqual(result['risk_scores']['scope'], 0)
            for rule in result['rule_matches']:
                self.assertEqual(text[rule['char_start']-37:rule['char_end']-37], rule['quote'])

    def test_license_adjectives_and_unrelated_purpose_do_not_lower_scope(self):
        for text in [NETGEAR.replace('Solely for this purpose, ', ''),
                     'Solely for this purpose, Supplier maintains records. Owner grants Customer a limited right to use Software.',
                     'Owner grants Customer a right to use Software. Solely for this purpose, Customer shall pay fees.',
                     'Solely for this purpose, Supplier pays and grants Customer a right to use Software.',
                     'Solely for this purpose, Supplier may request Owner to grant Customer a license to use Software.',
                     'The limited license is royalty-free and non-exclusive.']:
            with self.subTest(text=text):
                self.assertEqual(score(text, 'License Grant')['signal_sources']['scope'], 'profile_default_unverified')

    def test_mutual_variants_and_notice_lengths_only_affect_balance_support(self):
        for text in ['Either party may terminate this Agreement for convenience.',
                     'Each party may terminate the agreement without cause.',
                     MTI.replace('sixty (60)', 'ninety (90)'),
                     MTI.replace('sixty (60)', '30'), MTI.replace(' ', '\n  ')]:
            result = score(text, 'Termination For Convenience')
            self.assertEqual(result['signal_sources']['imbalance'], 'wording_rule_provisional')
            self.assertEqual(result['risk_total'], 3)

    def test_unilateral_unrelated_and_unequal_rights_not_mutual(self):
        for text in [MTI.replace('Either party', 'Supplier'),
                     'Either party may receive notices. Supplier may terminate this Agreement without cause.',
                     'Either party may terminate this Agreement without cause, but Supplier must give sixty days notice and Customer must give ten days notice.',
                     'Either party may terminate negotiations without cause.',
                     'Either party may terminate this Agreement for material breach.']:
            with self.subTest(text=text):
                self.assertEqual(score(text, 'Termination For Convenience')['signal_sources']['imbalance'], 'profile_default_unverified')

    def test_uncertainty_and_incomplete_evidence_preserved(self):
        for text, category, signal in [(NETGEAR, 'License Grant', 'scope'), (MTI, 'Termination For Convenience', 'imbalance')]:
            for modified in ['If approved, ' + text, 'It is not true that ' + text,
                             text + ' However, exceptions apply under Section 4.',
                             'Except as provided in Section 4. ' + text]:
                result = score(modified, category)
                self.assertFalse(result['usable_for_scoring'])
                self.assertEqual(result['signal_sources'][signal], 'profile_default_unverified')
            result = score_evidence(dict(category=category, original_text=text, char_start=0,
                                        char_end=len(text), localization='heuristic_passage',
                                        usable_for_scoring=False, review_reasons=['incomplete_right_context']))
            self.assertFalse(result['usable_for_scoring'])
            self.assertEqual(result['signal_sources'][signal], 'profile_default_unverified')

    def test_other_categories_do_not_apply_new_rules(self):
        for text in [NETGEAR, MTI]:
            result = score(text, 'Audit Rights')
            self.assertFalse(any(r['rule_id'].endswith((':license_purpose_restricted', ':exit_either_party')) for r in result['rule_matches']))


if __name__ == '__main__':
    unittest.main()
