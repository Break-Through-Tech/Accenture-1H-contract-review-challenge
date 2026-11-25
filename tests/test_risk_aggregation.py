"""Synthetic aggregation checks."""
import copy
import unittest

from risk_scoring.aggregation import aggregate_contracts
from risk_scoring.scoring import score_clauses, risk_level
from risk_scoring.evidence import extract_evidence
import test_risk_evidence


def finding(start=0, end=10, total=3, category='Audit Rights', contract='A'):
    remaining = total
    scores = {}
    for signal in ('severity', 'scope', 'imbalance', 'exposure'):
        scores[signal] = min(2, remaining)
        remaining -= scores[signal]
    return {'contract_id': contract, 'title': contract, 'category': category,
            'char_start': start, 'char_end': end, 'original_text': 'x' * (end-start),
            'risk_scores': scores, 'risk_total': total, 'risk_level': risk_level(total),
            'localization': 'heuristic_passage', 'supporting_chunk_ids': [0],
            'needs_review': True, 'review_reasons': ['provisional_rubric'],
            'rule_matches': [{'rule_id': 'synthetic-rule'}]}


class AggregationTests(unittest.TestCase):
    def test_distinct_groups_max_and_counts(self):
        rows = [finding(total=1), finding(20, 30, 4), finding(40, 50, 7), finding(60, 70, 6)]
        result = aggregate_contracts(rows)[0]
        self.assertEqual(result['triage_score'], 7)
        self.assertEqual(result['risk_level'], 'High')
        self.assertEqual(result['evidence_group_counts'], {'High': 2, 'Medium': 1, 'Low': 1})

    def test_duplicates_and_multiple_categories(self):
        row = finding(total=4)
        result = aggregate_contracts([row, copy.deepcopy(row), finding(total=6, category='Insurance')])[0]
        self.assertEqual(result['evidence_group_count'], 1)
        self.assertEqual(result['triage_score'], 6)
        self.assertEqual(result['evidence_groups'][0]['categories'], ['Audit Rights', 'Insurance'])
        self.assertEqual(len(result['clause_findings']), 3)

    def test_overlap_chain_and_adjacent_spans(self):
        result = aggregate_contracts([finding(0, 10), finding(8, 20), finding(18, 30), finding(30, 40)])[0]
        self.assertEqual(result['evidence_group_count'], 2)
        self.assertIn('overlapping_passages_grouped_provisionally', [flag['reason'] for flag in result['review_flags']])

    def test_contracts_without_evidence(self):
        results = aggregate_contracts([], contract_ids=['A', 'B'])
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsNone(result['triage_score'])
            self.assertEqual(result['risk_level'], 'Review Required')
            self.assertEqual(result['evidence_group_count'], 0)
        self.assertEqual(aggregate_contracts([]), [])

    def test_fallback_does_not_bridge_or_raise_score(self):
        fallback = finding(0, 100, 8)
        fallback['localization'] = 'chunk_fallback'
        result = aggregate_contracts([finding(0, 10, 1), finding(20, 30, 3), fallback])[0]
        self.assertEqual(result['triage_score'], 3)
        self.assertEqual(result['evidence_group_count'], 2)
        self.assertEqual(result['excluded_finding_indices'], [2])
        self.assertEqual(aggregate_contracts([fallback])[0]['risk_level'], 'Review Required')

    def test_unusable_scores_remain_traceable(self):
        for field, value in [('risk_total', None), ('risk_total', 9), ('risk_level', 'High'),
                             ('risk_scores', {}), ('char_end', 0), ('original_text', '')]:
            row = finding()
            row[field] = value
            result = aggregate_contracts([row])[0]
            self.assertEqual(result['risk_level'], 'Review Required')
            self.assertEqual(result['clause_findings'], [row])

    def test_review_flags_never_add_penalties(self):
        result = aggregate_contracts([finding(total=0)], review_flags={'A': ['Insurance not detected; applicability unknown']})[0]
        self.assertEqual(result['triage_score'], 0)
        self.assertTrue(result['needs_review'])
        self.assertIn('Insurance not detected; applicability unknown', [flag['reason'] for flag in result['review_flags']])

    def test_separate_contracts_and_order(self):
        rows = [finding(contract='B', total=8), finding(contract='A', total=1), finding(20, 30, 4, contract='A')]
        first, second = aggregate_contracts(rows), aggregate_contracts(reversed(rows))
        for left, right in zip(first, second):
            for key in ('contract_id', 'triage_score', 'risk_level', 'evidence_group_counts'):
                self.assertEqual(left[key], right[key])

    def test_preserves_explanations_without_mutation(self):
        rows = [finding()]
        original = copy.deepcopy(rows)
        result = aggregate_contracts(rows)[0]
        self.assertEqual(rows, original)
        self.assertEqual(result['clause_findings'][0]['rule_matches'], rows[0]['rule_matches'])
        result['clause_findings'][0]['rule_matches'].clear()
        self.assertEqual(rows, original)

    def test_conflicting_text_and_bad_identity_fail(self):
        other = finding()
        other['original_text'] = 'y' * 10
        with self.assertRaises(ValueError):
            aggregate_contracts([finding(), other])
        with self.assertRaises(ValueError):
            aggregate_contracts([{**finding(), 'contract_id': ''}])

    def test_all_bucket_boundaries(self):
        for total in range(9):
            result = aggregate_contracts([finding(total=total)])[0]
            self.assertEqual(result['risk_level'], risk_level(total))

    def test_synthetic_pipeline(self):
        fixture = test_risk_evidence.EvidenceTests()
        fixture.setUp()
        chunks = fixture.joined(text='Intro. Customer may inspect Supplier records; audit costs are unlimited. End.')
        scored = score_clauses(extract_evidence(chunks, fixture.categories))
        result = aggregate_contracts(scored, contract_ids=['contract-A', 'no-detections'])
        self.assertEqual(result[0]['triage_score'], 4)
        self.assertEqual(result[0]['clause_findings'], scored)
        self.assertEqual(result[1]['risk_level'], 'Review Required')


if __name__ == '__main__':
    unittest.main()
