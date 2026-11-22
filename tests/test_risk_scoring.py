"""Synthetic rule checks, not human-reviewed risk labels."""
import copy
import unittest

from risk_scoring.io import load_categories
from risk_scoring.scoring import load_profiles, score_evidence, score_clauses, risk_level, SCOPE_CUES, RULES
from risk_scoring.evidence import extract_evidence
import test_risk_evidence


def evidence(text, category='Audit Rights', probability=.8):
    return {'evidence_id': 'synthetic', 'contract_id': 'example', 'title': 'example',
            'category': category, 'original_text': text, 'char_start': 10,
            'char_end': 10 + len(text), 'localization': 'heuristic_passage',
            'supporting_chunk_ids': [0], 'max_detection_probability': probability,
            'supporting_predictions': [{'chunk_id': 0, 'probability': probability}],
            'needs_review': True, 'review_reasons': ['heuristic_boundaries']}


LOW = 'The parties audit only records relating to this agreement. The sole consequence of audit failure is a written reminder. Each party has identical audit rights and obligations.'
HIGH = 'The parties audit all books and records. Audit costs are unlimited. One party alone may expand the audit scope. Audit failure triggers termination of all services.'


class ScoringTests(unittest.TestCase):
    def test_same_category_different_scores(self):
        low, high = score_evidence(evidence(LOW)), score_evidence(evidence(HIGH))
        self.assertEqual(low['risk_scores'], dict(severity=0, scope=0, imbalance=0, exposure=1))
        self.assertEqual((low['risk_total'], low['risk_level']), (1, 'Low'))
        self.assertEqual((high['risk_total'], high['risk_level']), (8, 'High'))

    def test_all_profiles_and_rule_bounds(self):
        profiles = load_profiles()
        self.assertEqual(set(profiles['profiles']), set(load_categories()))
        for category in load_categories():
            result = score_evidence(evidence('Details remain unspecified.', category))
            self.assertEqual(result['risk_scores'], profiles['profiles'][category]['initial_scores'])
            self.assertTrue(all(0 <= value <= 2 for value in result['risk_scores'].values()))
            self.assertEqual(set(result['signal_sources'].values()), {'profile_default_unverified'})
        for category, patterns in SCOPE_CUES.items():
            for pattern, expected in zip(patterns, [2, 0]):
                self.assertEqual(score_evidence(evidence(pattern, category))['risk_scores']['scope'], expected)
        for _, categories, signal, pattern, value, _ in RULES:
            self.assertIn(signal, ('severity', 'scope', 'imbalance', 'exposure'))
            self.assertIn(value, (0, 1, 2))
            self.assertTrue(set(categories) <= set(load_categories()))

    def test_bucket_boundaries(self):
        self.assertEqual([risk_level(i) for i in range(9)], ['Low'] * 3 + ['Medium'] * 3 + ['High'] * 3)
        for value in [-1, 9, 1.5, True]:
            with self.assertRaises(ValueError):
                risk_level(value)

    def test_negation_exceptions_and_conditions_defer(self):
        for text in ['Audit costs are not unlimited.', 'It is not true that audit costs are unlimited.',
                     'Audit costs are unlimited except for routine audits.',
                     'Audit costs are unlimited unless otherwise agreed.',
                     'Audit costs are unlimited, subject to Section 4.',
                     'If requested, audit costs are unlimited.',
                     'Audit costs might be unlimited.']:
            result = score_evidence(evidence(text))
            self.assertEqual(result['risk_total'], 3)
            self.assertIn('negation_condition_or_reference_unresolved', result['review_reasons'])
            self.assertFalse(any(m['status'] == 'applied' for m in result['rule_matches']))

    def test_conflicting_rules_retain_default(self):
        text = 'Audit costs are unlimited. Audit costs are limited to a fixed fee.'
        result = score_evidence(evidence(text))
        self.assertEqual(result['risk_scores']['exposure'], 1)
        self.assertIn('conflicting_rules:exposure', result['review_reasons'])
        self.assertEqual({m['status'] for m in result['rule_matches']}, {'deferred_conflict'})

    def test_confidence_and_party_names_do_not_change_risk(self):
        first = score_evidence(evidence('Alpha and Beta agree. ' + HIGH, probability=.51))
        second = score_evidence(evidence('Beta and Alpha agree. ' + HIGH, probability=.99))
        self.assertEqual(first['risk_scores'], second['risk_scores'])
        self.assertEqual(first['max_detection_probability'], .51)
        self.assertEqual(second['max_detection_probability'], .99)

    def test_quotes_offsets_and_no_mutation(self):
        record = evidence(HIGH)
        before = copy.deepcopy(record)
        result = score_evidence(record)
        self.assertEqual(record, before)
        for match in result['rule_matches']:
            self.assertEqual(record['original_text'][match['char_start'] - 10:match['char_end'] - 10], match['quote'])
        self.assertEqual(result['supporting_predictions'], record['supporting_predictions'])
        self.assertEqual(score_evidence(evidence(HIGH + ' ' + HIGH))['risk_total'], result['risk_total'])

    def test_fallback_does_not_apply_wording(self):
        record = evidence(HIGH)
        record['localization'] = 'chunk_fallback'
        result = score_evidence(record)
        self.assertEqual(result['risk_total'], 3)
        self.assertTrue(result['needs_review'])

    def test_category_specificity(self):
        result = score_evidence(evidence(HIGH, 'Document Name'))
        self.assertEqual(result['risk_total'], 0)
        self.assertEqual(result['rule_matches'], [])
        self.assertFalse(result['needs_review'])
        self.assertIn('no_wording_rule_matched', result['limitations'])
        self.assertEqual(result['risk_status'], 'provisional_heuristic_not_human_reviewed')

    def test_extractor_integration_synthetic(self):
        fixture = test_risk_evidence.EvidenceTests()
        fixture.setUp()
        chunks = fixture.joined(text='Intro. Audit costs are unlimited. End.')
        records = extract_evidence(chunks, fixture.categories)
        result = score_clauses(records)
        self.assertEqual(result[0]['risk_scores']['exposure'], 2)
        self.assertEqual(result[0]['evidence_id'], records[0]['evidence_id'])

    def test_invalid_evidence(self):
        for record in [evidence(''), {**evidence(HIGH), 'char_end': 0}]:
            with self.assertRaises(ValueError):
                score_evidence(record)


if __name__ == '__main__':
    unittest.main()
