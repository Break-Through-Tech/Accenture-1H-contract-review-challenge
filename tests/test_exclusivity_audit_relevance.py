"""Focused category-relevance checks; no human risk labels are inferred."""
import unittest

from test_contextual_scoring import score
from test_evidence_quality import extract
from risk_scoring.scoring import score_clauses
from risk_scoring.aggregation import aggregate_contracts
from risk_scoring.run import assessment_fields

NETGEAR = "Following the initial one (1) year term, for any extension or renewal term, Distributor's appointment as Distributor shall be non-exclusive and NETGEAR may appoint other distributors in the Territory at its sole discretion."
PLAYBOY = 'All EFS Produced Merchandise, co-branded Merchandise, mass-customized Merchandise, exclusive Merchandise and Client Inventory shall be deemed "Merchandise" for purposes of the calculation of Royalties (defined below) to be paid by EFS.'
BELLRING = '(c) Heritage shall perform, and shall ensure Jasper performs, a documented inspection of all trailers before loading to confirm they are free of any visible contamination or odors and fit for use with food products.'
INMODE = 'That includes full annually audited financials statements and, subjected the credit limit analysis request, Quarterly financial statements (P&L, BS and Cash Flow statements).'
KALLO = "Kallo shall permit Agent or its designated Agent(s) to inspect Kallo's books upon Agent's request at Agent's expense during normal business hours of Kallo."


class ExclusivityAuditRelevanceTests(unittest.TestCase):
    def test_named_nonoperative_findings_are_deferred_not_zeroed(self):
        for category, text in [('Exclusivity', NETGEAR), ('Exclusivity', PLAYBOY),
                               ('Audit Rights', BELLRING), ('Audit Rights', INMODE)]:
            result = score(text, category)
            self.assertEqual(result['original_text'], text)
            self.assertEqual(result['risk_total'], 3)
            self.assertFalse(result['usable_for_scoring'])
            self.assertIn('category_relevant_provision_not_established', result['review_reasons'])
            self.assertEqual(result['numeric_score_status'], 'deferred_review')
            self.assertIsNone(aggregate_contracts([result])[0]['triage_score'])
            self.assertEqual(assessment_fields(result)['assessment_status'], 'human_review_required')

    def test_ultragenyx_and_kallo_remain_eligible(self):
        for category, text in [('Exclusivity', 'Cremer shall supply Ultragenyx exclusively with the Product worldwide.'),
                               ('Exclusivity', 'Ultragenyx shall purchase the Product exclusively from Cremer.'),
                               ('Audit Rights', KALLO)]:
            result = score(text, category)
            self.assertTrue(result['usable_for_scoring'])
            self.assertEqual(result['risk_total'], 3)

    def test_mixed_exclusivity_is_retained(self):
        for text in ['Distributor is exclusive for the first year. Its appointment shall be non-exclusive thereafter.',
                     'Supplier shall supply Buyer exclusively. Reseller receives a non-exclusive license.',
                     PLAYBOY + ' Buyer shall purchase these goods exclusively from Seller.']:
            result = score(text, 'Exclusivity')
            self.assertTrue(result['usable_for_scoring'])
        result = score('Supplier shall sell exclusively to Buyer except for named customers. Distributor is non-exclusive elsewhere.', 'Exclusivity')
        self.assertNotIn('category_relevant_provision_not_established', result['review_reasons'])
        self.assertFalse(result['usable_for_scoring'])  # Existing exception safeguard.

    def test_audit_permissions_and_passive_wording(self):
        for text in ['Customer may audit Supplier records to verify compliance.',
                     'Supplier agrees to allow Customer representatives to inspect its facilities.',
                     'Supplier records may be inspected by Customer.',
                     'Such records will be open for inspection by Customer.',
                     "The Licensee agrees to allow the Licensor to conduct reasonable inspection of the quality of the Licensee's services from time to time.",
                     "The audit rights include the right to have the Auditor verify Ono's compliance.",
                     'Manufacturer may review routine reports relating to nonconforming materials identified by Subcontractor.',
                     'Administrator shall provide copies of all historical records as requested to enable the Trust or its representatives to verify compliance by Administrator with this Agreement.',
                     'Customer has the right to examine Supplier books.']:
            with self.subTest(text=text):
                self.assertTrue(score(text, 'Audit Rights')['usable_for_scoring'])

    def test_incidental_and_internal_inspections_are_not_counterparty_rights(self):
        for text in ['Supplier shall inspect all trailers before loading.',
                     'Supplier may inspect its own records.',
                     'Customer may receive audited financial statements.',
                     'Supplier shall maintain accurate books and records.']:
            self.assertFalse(score(text, 'Audit Rights')['usable_for_scoring'])

    def test_adjacent_permission_recovery_preserves_offsets_and_probability(self):
        lead = 'Supplier shall maintain books and records.'
        text = 'Intro. ' + lead + ' Customer may audit Supplier records.'
        rows = extract(text, 'Audit Rights', split=len('Intro. ' + lead))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['original_text'], text[7:])
        self.assertEqual((rows[0]['char_start'], rows[0]['char_end']), (7, len(text)))
        self.assertEqual(rows[0]['supporting_chunk_ids'], [0])
        self.assertTrue(score_clauses(rows)[0]['usable_for_scoring'])

    def test_audit_heading_recovers_records_and_adjacent_permission(self):
        heading = 'Audit Rights.'
        text = heading + ' Supplier shall keep books and records. Customer may inspect such records.'
        rows = extract(text, 'Audit Rights', split=len(heading))
        self.assertEqual(rows[0]['original_text'], text)
        self.assertEqual(rows[0]['char_end'], len(text))
        self.assertTrue(score_clauses(rows)[0]['usable_for_scoring'])

    def test_recovery_does_not_skip_headings_or_unrelated_text(self):
        lead = 'Supplier shall maintain books and records.'
        for middle in ['Payment Terms.', 'Supplier shall ship goods.']:
            text = lead + ' ' + middle + ' Customer may audit Supplier records.'
            rows = extract(text, 'Audit Rights', split=len(lead))
            self.assertEqual(rows[0]['original_text'], lead)
            self.assertFalse(score_clauses(rows)[0]['usable_for_scoring'])

    def test_negation_exception_and_incomplete_recovery_stay_deferred(self):
        for text in ['Customer may not audit Supplier records.',
                     'Customer may audit Supplier records except as provided in Section 4.']:
            self.assertFalse(score(text, 'Audit Rights')['usable_for_scoring'])
        lead = 'Supplier shall maintain books and records.'
        for tail in ['Customer may audit Supplier records',
                     'Customer may audit Supplier records. However, exceptions apply under Section 4.']:
            self.assertFalse(score_clauses(extract(lead + ' ' + tail, 'Audit Rights', split=len(lead)))[0]['usable_for_scoring'])

    def test_internal_records_with_or_without_own_are_deferred(self):
        for phrase in ['its records', 'its own records', 'their records', 'their own records']:
            self.assertFalse(score('Supplier may inspect ' + phrase + '.', 'Audit Rights')['usable_for_scoring'])

    def test_internal_inspection_does_not_hide_separate_customer_permission(self):
        for internal in ['Supplier may inspect its records', 'Supplier may inspect its own records']:
            text = internal + ' and shall permit Customer to audit Supplier records.'
            rows = extract(text, 'Audit Rights')
            result = score_clauses(rows)[0]
            self.assertTrue(result['usable_for_scoring'])
            self.assertEqual(result['risk_total'], 3)
            self.assertEqual(result['original_text'], text)
            self.assertEqual((result['char_start'], result['char_end']), (0, len(text)))
            self.assertFalse(score(text + ' Except as provided in Section 4.', 'Audit Rights')['usable_for_scoring'])
        self.assertTrue(score('Supplier shall permit Customer to audit its records.', 'Audit Rights')['usable_for_scoring'])

    def test_nonexclusive_whitespace_variations_do_not_establish_commitment(self):
        for term in ['non-exclusive', 'non-\nexclusive', 'non-  exclusive', 'non -\n exclusive', 'non\nexclusive']:
            text = 'Distributor shall be ' + term + '.'
            result = score(text, 'Exclusivity')
            self.assertFalse(result['usable_for_scoring'])
            self.assertEqual(result['original_text'], text)

    def test_present_tense_exclusive_purchase_with_separate_license(self):
        for term in ['non-exclusive', 'non-\nexclusive']:
            text = 'Buyer purchases Products exclusively from Seller. Reseller receives a ' + term + ' license.'
            result = score(text, 'Exclusivity')
            self.assertTrue(result['usable_for_scoring'])
            self.assertEqual(result['risk_total'], 3)
            self.assertEqual(result['original_text'], text)
            self.assertFalse(score(text.replace('purchases', 'does not purchase'), 'Exclusivity')['usable_for_scoring'])
            self.assertFalse(score(text + ' Except for named products.', 'Exclusivity')['usable_for_scoring'])


if __name__ == '__main__':
    unittest.main()
