"""Contract triage from distinct groups of clause evidence."""
from copy import deepcopy
import hashlib

from .scoring import SIGNALS, INFORMATIONAL, risk_level


def _unusable_reason(finding):
    if finding.get('localization') != 'heuristic_passage' or finding.get('usable_for_scoring') is False:
        return 'unlocalized_evidence_excluded'
    start, end = finding.get('char_start'), finding.get('char_end')
    text = finding.get('original_text')
    if (type(start) is not int or type(end) is not int or start < 0 or end <= start
            or not isinstance(text, str) or not text.strip() or len(text) != end - start):
        return 'invalid_source_span'
    scores = finding.get('risk_scores')
    if (not isinstance(scores, dict) or set(scores) != set(SIGNALS)
            or any(type(value) is not int or not 0 <= value <= 2 for value in scores.values())):
        return 'missing_or_invalid_signal_scores'
    total = finding.get('risk_total')
    if type(total) is not int or total != sum(scores.values()):
        return 'missing_or_inconsistent_total'
    if finding.get('risk_level') != risk_level(total):
        return 'inconsistent_risk_level'
    return None


def aggregate_contracts(scored_findings, contract_ids=None, review_flags=None):
    """Return contract summaries with groups, original findings, and review flags.

    Pass the full contract list to include contracts with no detections.
    review_flags maps contract IDs to externally supplied review notes.
    """
    review_flags = review_flags or {}
    contracts = {}
    contract_ids = [] if contract_ids is None else list(contract_ids)
    for contract_id in contract_ids + list(review_flags):
        if not isinstance(contract_id, str) or not contract_id:
            raise ValueError('Contract IDs must be nonempty strings')
        contracts.setdefault(contract_id, [])
    for finding in scored_findings:
        contract_id = finding.get('contract_id')
        if not isinstance(contract_id, str) or not contract_id:
            raise ValueError('Each finding needs a contract_id')
        if finding.get('title', contract_id) != contract_id:
            raise ValueError('Contract ID and title disagree')
        contracts.setdefault(contract_id, []).append(deepcopy(finding))
    summaries = []
    for contract_id, findings in sorted(contracts.items()):
        notes = review_flags.get(contract_id, [])
        if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
            raise ValueError('Review flags must be lists of strings')
        flags = [{'finding_index': None, 'reason': note} for note in sorted(set(notes))]
        usable = []
        for index, finding in enumerate(findings):
            reasons = set(finding.get('review_reasons', []))
            if finding.get('needs_review'):
                reasons.add('clause_requires_review')
            unusable = _unusable_reason(finding)
            if unusable:
                reasons.add(unusable)
            else:
                usable.append(index)
            flags.extend({'finding_index': index, 'reason': reason} for reason in sorted(reasons))
        groups = []
        for index in sorted(usable, key=lambda i: (findings[i]['char_start'], findings[i]['char_end'])):
            finding = findings[index]
            start, end = finding['char_start'], finding['char_end']
            if groups and start < groups[-1]['char_end']:
                group = groups[-1]
                # Overlapping source text must agree, even across categories.
                for previous_index in group['finding_indices']:
                    previous = findings[previous_index]
                    overlap_start = max(start, previous['char_start'])
                    overlap_end = min(end, previous['char_end'])
                    if overlap_start < overlap_end:
                        current_text = finding['original_text'][overlap_start-start:overlap_end-start]
                        previous_text = previous['original_text'][overlap_start-previous['char_start']:overlap_end-previous['char_start']]
                        if current_text != previous_text:
                            raise ValueError(f'Conflicting source text in contract {contract_id}')
                group['char_end'] = max(group['char_end'], end)
                group['finding_indices'].append(index)
            else:
                groups.append({'char_start': start, 'char_end': end, 'finding_indices': [index]})
        counts = {'High': 0, 'Medium': 0, 'Low': 0}
        for group in groups:
            members = [findings[index] for index in group['finding_indices']]
            score = max(member['risk_total'] for member in members)
            level = risk_level(score)
            identity = f"{contract_id}\0{group['char_start']}\0{group['char_end']}"
            group.update({
                'group_id': hashlib.sha256(identity.encode()).hexdigest()[:24],
                'categories': sorted({member['category'] for member in members}),
                'risk_score': score, 'risk_level': level,
                'substantive_evidence': any(member.get('substantive_evidence', member['category'] not in INFORMATIONAL and member['category'] != 'Competitive Restriction Exception') for member in members),
                'highest_score_finding_indices': [index for index in group['finding_indices'] if findings[index]['risk_total'] == score],
                'supporting_chunk_ids': sorted({chunk for member in members for chunk in member.get('supporting_chunk_ids', [])}),
            })
            if len({(member['char_start'], member['char_end']) for member in members}) > 1:
                flags.append({'finding_index': None, 'group_id': group['group_id'],
                              'reason': 'overlapping_passages_grouped_provisionally'})
            counts[level] += 1
        primary_score = max((group['risk_score'] for group in groups if group['substantive_evidence']), default=None)
        if primary_score is None:
            flags.append({'finding_index': None, 'reason': 'no_usable_substantive_evidence'})
        summaries.append({
            'contract_id': contract_id,
            'triage_score': primary_score,
            'risk_level': risk_level(primary_score) if primary_score is not None else 'Review Required',
            'risk_status': 'provisional_heuristic_not_human_reviewed',
            'evidence_group_counts': counts, 'evidence_group_count': len(groups),
            'evidence_groups': groups, 'clause_findings': findings,
            'excluded_finding_indices': [index for index in range(len(findings)) if index not in usable],
            'needs_review': bool(flags), 'review_flags': flags,
        })
    return summaries
