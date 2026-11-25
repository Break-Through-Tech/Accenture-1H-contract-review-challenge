"""Extract candidate passages for review."""
import hashlib
import math
import re
from collections.abc import Mapping

from .cues import CUES

# Matches the training notebook; provisional until threshold provenance is verified.
DEFAULT_THRESHOLD = 0.5
CUE_VERSION = '0.12-provisional'
BOUNDARY = re.compile(r'(?<=[.!?])[ \t]+|\n[ \t]*\n+')
QUALIFIER = re.compile(r'\b(?:except|notwithstanding|provided that|subject to|unless|however|section|schedule|exhibit)\b', re.I)


OPERATIVE = re.compile(r"\b(?:shall|may|will|must|is|are|has|have|grants?|agrees?|acknowledges?|assigns?|pays?|provides?|maintains?|appl(?:y|ies)|means|includes?|constitutes?|continues?|permits?|covers?|audit(?=\s+(?:all|only|such|the|records))|inspect(?=\s+(?:all|only|such|the|records)))\b", re.I)


def is_assignment_heading(text):
    return bool(re.fullmatch(r'\s*(?:\d+(?:\.\d+)*\.?\s+)?successors\s+and\s+assigns[.:]?\s*', text, re.I))


def is_heading(text, category=None):
    if category == 'Anti-Assignment' and is_assignment_heading(text):
        return True
    if category in {'Anti-Assignment', 'Cap On Liability'} and re.search(r'\b(?:requires?|states|constitutes)\b', text, re.I):
        return False
    if re.fullmatch(r'\s*(?:\([a-z0-9]+\)\s*)?(?:additional\s+)?actions?\s+to\s+assign\s+interests?[.:]?\s*', text, re.I):
        return True
    return len(text.split()) <= 16 and not OPERATIVE.search(text)


def is_contents_entry(text):
    return not OPERATIVE.search(text) and bool(re.search(
        r'\.{3,}|\btable of contents\b|\b\d+\s+(?:Section|Article)\s+\d|[A-Za-z]\s+\d+\s*$', text, re.I))


def is_license_definition(text):
    return bool(re.search(
        r'^\s*(?:\d+(?:\.\d+)*\s+)?["\u201c]?[A-Za-z][\w -]{0,45}["\u201d]?\s+(?:shall\s+)?means\b|\bis\s+defined\s+as\b', text, re.I))


def license_grant_present(text):
    """Recognize an affirmative grant, not a definition or denial of one."""
    for left, right in _segments(text):
        sentence = text[left:right]
        if is_license_definition(sentence) or re.search(r'\b(?:nothing|not|no|never|neither)\b', sentence, re.I):
            continue
        if re.search(r'\bgrants?\b[^.!?;]{0,180}\b(?:rights?|licen[cs]e)\b', sentence, re.I):
            return True
    return False


def license_nonoperative_reason(text):
    if license_grant_present(text):
        return None
    if re.search(r'\b(?:nothing|no|not|never|neither)\b[^.!?;]{0,140}\bgrant\w*\b', text, re.I):
        return 'license_grant_denied_not_established'
    if any(is_license_definition(text[left:right]) for left, right in _segments(text)):
        return 'license_definition_without_grant'
    return None


def audit_permission_present(text):
    """Require permission plus an audit action and its object, not 'audit' alone.

    This is relevance evidence only; negation and exceptions still go through
    the scorer's existing uncertainty checks.
    """
    action = r'(?:audit|inspect|examine|review|verify|conduct\s+(?:reasonable\s+)?inspection)'
    objects = r'(?:books|records|accounts|premises|facilities|locations?|compliance|procedures|equipment|expenses|program|reports|quality|services|performance)'
    for left, right in _segments(text):
        sentence = text[left:right]
        # Ignore only the internal inspection phrase, not other permissions in
        # the sentence. This temporary mask never changes exported source text.
        sentence = re.sub(
            r'\bmay\s+' + action + r'\s+(?:its|their)\s+(?:own\s+)?(?:books|records|premises|facilities|trailers)\b',
            lambda match: ' ' * len(match.group()), sentence, flags=re.I)
        if re.search(
                r'\b(?:may|right\s+to|shall\s+permit|agrees?\s+to\s+allow|allow)\b[^.!?;]{0,120}\b' + action
                + r'\b[^.!?;]{0,140}\b' + objects + r'\b|'
                r'\b' + objects + r'\b[^.!?;]{0,100}\b(?:may\s+be\s+(?:inspected|audited)|shall\s+be\s+open\s+for\s+inspection|will\s+be\s+open\s+for\s+inspection)\b|'
                r'\bparties\s+' + action + r'\s+(?:all|only|the)\s+(?:books|records)\b', sentence, re.I):
            return True
        # Access to specified records expressly for the counterparty's review
        # is also permission; merely supplying financial statements is not.
        if re.search(r'\bshall\s+provide\s+copies\b[^.!?;]{0,120}\brecords\b[^.!?;]{0,450}\benable\b[^.!?;]{0,180}\b(?:representatives|auditors)\b[^.!?;]{0,300}\bverify\s+compliance\b', sentence, re.I):
            return True
    return False


def following_provision(text, start, end, source, chunk_start, category=None, require_grant=False, require_audit=False):
    extended = text
    for offset in range(chunk_start + len(text), chunk_start + len(text) + 2000):
        if offset not in source:
            break
        extended += source[offset]
    following = [(end + a, end + b) for a, b in _segments(extended[end:])
                 if extended[end + a:end + b].strip()]
    if not following:
        return end, False
    next_start, next_end = following[0]
    passage = extended[next_start:next_end].strip()
    if is_contents_entry(passage) or is_heading(passage, category):
        return end, False
    if require_grant and not license_grant_present(passage):
        return end, False
    if require_audit and not audit_permission_present(passage):
        return end, False
    recovered_end = next_end
    if category == 'Audit Rights' and not audit_permission_present(passage) and len(following) > 1:
        left, right = following[1]
        adjacent = extended[left:right].strip()
        if not is_heading(adjacent, category) and audit_permission_present(adjacent):
            recovered_end = right
    # Successor boilerplate often separates this heading from the transfer rule.
    # Recover only the immediately adjacent assignment provision, never scan
    # past another heading or unrelated text to manufacture category support.
    if (category == 'Anti-Assignment' and is_assignment_heading(text[start:end])
            and re.search(r'\bsuccessors\s+and\s+(?:permitted\s+)?assigns\b', passage, re.I)
            and len(following) > 1):
        left, right = following[1]
        adjacent = extended[left:right].strip()
        if (not is_heading(adjacent, category) and not is_contents_entry(adjacent)
                and re.search(r'\bassign(?:ment|ed)?\b', adjacent, re.I)):
            recovered_end = right
    for next_start, next_end in following[1:]:
        if next_end <= recovered_end:
            continue
        passage = extended[next_start:next_end].strip()
        if not QUALIFIER.search(passage):
            break
        qualification = re.match(r'(?:except|unless|however|notwithstanding|provided that|subject to)\b', passage, re.I)
        if is_contents_entry(passage) or (is_heading(passage, category) and not qualification):
            break
        recovered_end = next_end
    complete = extended[:recovered_end].rstrip().endswith(('.', '!', '?'))
    # A full buffer ending mid-provision is not a reliable clause boundary.
    if recovered_end == len(extended) and chunk_start + len(extended) in source:
        complete = False
    return recovered_end, complete


def _segments(text):
    start = 0
    for match in BOUNDARY.finditer(text):
        if re.search(r'\b(?:Inc|Ltd|Corp|Co|Mr|Mrs|Dr|e\.g|i\.e)\.$', text[:match.start()], re.I):
            continue
        if match.start() > start:
            yield start, match.start()
        start = match.end()
    if start < len(text):
        yield start, len(text)


def _thresholds(categories, threshold):
    if isinstance(threshold, Mapping):
        if set(threshold) != set(categories):
            raise ValueError('Threshold mapping must cover every category exactly')
        values = dict(threshold)
    else:
        values = dict.fromkeys(categories, threshold)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError('Thresholds must be finite numbers in [0, 1]')
    return values


def _source_maps(frame):
    """Check overlapping text and retain original character positions."""
    source_maps = {}
    for title, chunks in frame.groupby('title', sort=True):
        source = {}
        for row in chunks.itertuples(index=False):
            for offset, char in enumerate(row.chunk_text, int(row.char_start)):
                if offset in source and source[offset] != char:
                    raise ValueError(f'Inconsistent overlapping source text: {title}')
                source[offset] = char
        source_maps[title] = source
    return source_maps


def extract_evidence(frame, categories, threshold=DEFAULT_THRESHOLD,
                     threshold_status='provisional', threshold_source='configurable default; training notebook uses 0.5'):
    """Extract reviewable passages from the validated loader output."""
    if set(categories) != set(CUES):
        raise ValueError('Localization cues must match categories exactly')
    thresholds = _thresholds(categories, threshold)
    sources = _source_maps(frame)
    candidates = []
    for row in frame.sort_values(['title', 'chunk_id']).to_dict('records'):
        text = row['chunk_text']
        segments = list(_segments(text))
        for category in categories:
            probability = float(row[category])
            if probability < thresholds[category]:
                continue
            cue = re.compile(CUES[category], re.I)
            matches = [index for index, (start, end) in enumerate(segments) if cue.search(text[start:end])]
            spans = []
            for index in matches:
                start, end = segments[index]
                passage = text[start:end]
                if is_contents_entry(passage):
                    localization = 'table_of_contents'
                elif category == 'License Grant' and license_nonoperative_reason(passage):
                    end, complete = following_provision(text, start, end, sources[row['title']], int(row['char_start']), category, require_grant=True)
                    localization = ('heuristic_passage' if complete else 'incomplete_provision') if end > segments[index][1] else 'heuristic_passage'
                elif is_heading(passage, category):
                    end, complete = following_provision(text, start, end, sources[row['title']], int(row['char_start']), category)
                    localization = ('heuristic_passage' if complete else 'incomplete_provision') if end > segments[index][1] else 'heading_only'
                elif category == 'Audit Rights' and not audit_permission_present(passage):
                    end, complete = following_provision(text, start, end, sources[row['title']], int(row['char_start']), category, require_audit=True)
                    localization = ('heuristic_passage' if complete else 'incomplete_provision') if end > segments[index][1] else 'heuristic_passage'
                else:
                    localization = 'heuristic_passage'
                if localization == 'heuristic_passage' and end == segments[index][1] and index + 1 < len(segments):
                    next_start, next_end = segments[index + 1]
                    if QUALIFIER.search(text[next_start:next_end]):
                        end = next_end
                spans.append((start, end, localization))
            if not spans:
                spans = [(0, len(text), 'chunk_fallback')]
            for start, end, localization in spans:
                reasons = set()
                if localization != 'heuristic_passage':
                    reasons.add({'chunk_fallback': 'no_localization_cue', 'heading_only': 'heading_without_provision', 'table_of_contents': 'table_of_contents_entry', 'incomplete_provision': 'incomplete_recovered_provision'}[localization])
                span_text = ''.join(sources[row['title']][offset] for offset in range(int(row['char_start']) + start, int(row['char_start']) + end))
                if start == 0 and row['char_start'] > 0 and sources[row['title']].get(int(row['char_start']) - 1, '') not in '.!?\n':
                    reasons.add('incomplete_left_context')
                if end >= len(text) and not span_text.rstrip().endswith(('.', '!', '?')):
                    reasons.add('incomplete_right_context')
                if QUALIFIER.search(span_text):
                    reasons.add('qualification_or_reference_requires_review')
                candidates.append(dict(title=row['title'], category=category,
                    char_start=int(row['char_start']) + start, char_end=int(row['char_start']) + end,
                    localization=localization, reasons=reasons,
                    support={int(row['chunk_id']): probability}))
    # Merge overlapping spans, not adjacent or repeated text at different offsets.
    merged = []
    sort_fields = ('title', 'category', 'localization', 'char_start', 'char_end')
    for item in sorted(candidates, key=lambda candidate: tuple(candidate[field] for field in sort_fields)):
        if (merged and all(merged[-1][field] == item[field] for field in ('title', 'category', 'localization'))
                and item['char_start'] < merged[-1]['char_end']):
            previous = merged[-1]
            if (previous['char_start'], previous['char_end']) != (item['char_start'], item['char_end']):
                previous['reasons'].add('overlapping_candidate_boundaries_merged')
            previous['char_end'] = max(previous['char_end'], item['char_end'])
            previous['support'].update(item['support'])
            previous['reasons'].update(item['reasons'])
        else:
            merged.append(item)
    results = []
    for item in merged:
        title, category = item['title'], item['category']
        start, end = item['char_start'], item['char_end']
        source = sources[title]
        original_text = ''.join(source[index] for index in range(start, end))
        identity = f'{title}\0{category}\0{start}\0{end}\0{item["localization"]}'
        results.append({
            'evidence_id': hashlib.sha256(identity.encode()).hexdigest()[:24],
            'contract_id': title, 'title': title, 'category': category,
            'original_text': original_text, 'char_start': start, 'char_end': end,
            'supporting_chunk_ids': sorted(item['support']),
            'supporting_predictions': [{'chunk_id': chunk_id, 'probability': item['support'][chunk_id]} for chunk_id in sorted(item['support'])],
            'max_detection_probability': max(item['support'].values()),
            'probability_semantics': 'chunk-level category presence; not risk or calibrated passage confidence',
            'threshold': thresholds[category], 'threshold_status': threshold_status,
            'threshold_source': threshold_source, 'cue_version': CUE_VERSION,
            'localization': item['localization'],
            'usable_for_scoring': item['localization'] == 'heuristic_passage' and not ({'incomplete_left_context', 'incomplete_right_context'} & item['reasons']),
            'needs_review': bool(item['reasons']),
            'limitations': ['heuristic_localization_not_verified'],
            'review_reasons': sorted(item['reasons']),
        })
    # Fallback regions may duplicate localized passages; leave that decision to review.
    for item in results:
        if item['localization'] == 'chunk_fallback' and any(
            other['contract_id'] == item['contract_id'] and other['category'] == item['category']
            and other['localization'] == 'heuristic_passage'
            and max(item['char_start'], other['char_start']) < min(item['char_end'], other['char_end'])
            for other in results
        ):
            item['review_reasons'].append('may_duplicate_localized_evidence_do_not_count_separately')
    return results
