"""Focused validation regressions and synthetic positive counterexamples."""
import unittest

from test_contextual_scoring import score
from test_evidence_quality import extract
from risk_scoring.aggregation import aggregate_contracts
from risk_scoring.scoring import score_clauses


HEADING = 'Successors and Assigns.'
BOILERPLATE = 'This Agreement shall be binding upon and inure to the benefit of the parties and their successors and assigns.'
RESTRICTION = 'Distributor shall not assign this Agreement or any rights or obligations hereunder without the prior written consent of the Company.'
INSURANCE = "This insurance will include (a) worker's compensation insurance, (b) comprehensive general liability insurance, including coverage for product liability, bodily injury and property damage, and (c) automobile liability insurance."
REMEDY = "End-users' exclusive remedy is to receive replacement Product from reseller and NETGEAR's sole obligation and liability under this warranty is to issue an off-setting credit to reseller for Product returned by reseller on behalf of its end-user because of defects in workmanship or material."


class CategoryRelevanceTests(unittest.TestCase):
    def assert_excluded(self, rows):
        self.assertTrue(all(not row['usable_for_scoring'] for row in rows))
        self.assertTrue(all(row['needs_review'] for row in rows))
        self.assertTrue(all(row['numeric_score_status'] == 'deferred_review' for row in rows))
        self.assertIsNone(aggregate_contracts(rows)[0]['triage_score'])

    def test_blackbox_heading_and_boilerplate_cannot_score(self):
        for text in [HEADING, BOILERPLATE, HEADING + ' ' + BOILERPLATE]:
            with self.subTest(text=text):
                self.assert_excluded(score_clauses(extract(text, 'Anti-Assignment')))
                self.assert_excluded([score(text, 'Anti-Assignment')])
        self.assertEqual(extract(HEADING, 'Anti-Assignment')[0]['localization'], 'heading_only')

    def test_recover_blackbox_restriction_across_chunk_with_exact_offsets(self):
        text = 'Intro. ' + HEADING + ' ' + BOILERPLATE + ' ' + RESTRICTION
        rows = extract(text, 'Anti-Assignment', split=len('Intro. ' + HEADING))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row['char_start'], row['char_end']), (7, len(text)))
        self.assertEqual(row['original_text'], text[7:])
        self.assertEqual(row['supporting_chunk_ids'], [0])
        result = score_clauses(rows)[0]
        self.assertNotIn('category_relevant_provision_not_established', result['review_reasons'])
        # Recovered negation still goes through the existing uncertainty policy.
        self.assertIn('negation_condition_or_reference_unresolved', result['review_reasons'])

    def test_recovery_preserves_qualification_and_stops_at_heading(self):
        provision = 'Assignment requires prior written consent.'
        text = HEADING + ' ' + provision + ' Except for transfers to affiliates.'
        row = extract(text, 'Anti-Assignment', split=len(HEADING))[0]
        self.assertEqual(row['original_text'], text)
        self.assert_excluded(score_clauses([row]))
        text = HEADING + ' ' + BOILERPLATE + ' Payment Terms. ' + provision
        row = extract(text, 'Anti-Assignment', split=len(HEADING))[0]
        self.assertEqual(row['original_text'], HEADING + ' ' + BOILERPLATE)
        self.assert_excluded(score_clauses([row]))

    def test_operative_assignment_provisions_preserved(self):
        for text in ['Assignment requires prior written consent.',
                     'Any unauthorized assignment of this Agreement is void.',
                     'The parties will notify each other to reach further agreement regarding the assignment of rights and obligations.',
                     'Kallo may assign this Agreement to any party without the consent of Agent.',
                     'In any permitted assignment, Customer will remain liable for its obligations hereunder.']:
            with self.subTest(text=text):
                result = score(text, 'Anti-Assignment')
                self.assertTrue(result['usable_for_scoring'])
                self.assertEqual(aggregate_contracts([result])[0]['triage_score'], 3)

    def test_netgear_insurance_not_a_liability_cap(self):
        rows = score_clauses(extract(INSURANCE, 'Cap On Liability'))
        self.assert_excluded(rows)
        self.assertIn('category_relevant_provision_not_established', rows[0]['review_reasons'])
        self.assertEqual(rows[0]['original_text'], INSURANCE)

    def test_liability_caps_and_remedy_limits_preserved(self):
        for text in [REMEDY,
                     'Aggregate liability shall not exceed 10000 dollars.',
                     'Liability is capped at a fixed amount.',
                     "Cremer's liability arising from this Agreement is limited to intentional misconduct or gross negligence.",
                     'The foregoing states the entire liability of the parties concerning infringement.',
                     "Kallo's limitation of liability is cumulative with all expenditures aggregated to determine satisfaction of the limit."]:
            with self.subTest(text=text):
                result = score(text, 'Cap On Liability')
                self.assertTrue(result['usable_for_scoring'])
                self.assertIsNotNone(aggregate_contracts([result])[0]['triage_score'])

    def test_cap_heading_and_unrelated_obligation_do_not_establish_cap(self):
        for text in ['Limitation of Liability. The supplier shall deliver goods.',
                     'Supplier shall indemnify Customer against all claims and liabilities.',
                     'The supplier shall maintain general liability insurance limited to project operations.']:
            self.assert_excluded([score(text, 'Cap On Liability')])

    def test_insurance_plus_real_cap_is_preserved(self):
        result = score(INSURANCE + ' Aggregate liability shall not exceed 10000 dollars.', 'Cap On Liability')
        self.assertTrue(result['usable_for_scoring'])

    def test_negative_cap_construction_still_retains_relevant_evidence(self):
        result = score('In no event shall total liability exceed the fees paid.', 'Cap On Liability')
        self.assertNotIn('category_relevant_provision_not_established', result['review_reasons'])
        self.assertIn('negation_condition_or_reference_unresolved', result['review_reasons'])

    def test_other_categories_unaffected(self):
        for text, category in [
                ('The Developer hereby assigns all rights in deliverables to Client.', 'Ip Ownership Assignment'),
                ('The supplier shall maintain general liability insurance.', 'Insurance'),
                ('The owner grants a worldwide license.', 'License Grant')]:
            self.assertTrue(score(text, category)['usable_for_scoring'])


if __name__ == '__main__':
    unittest.main()
