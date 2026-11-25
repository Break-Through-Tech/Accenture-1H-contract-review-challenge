"""Provisional clause scores with traceable wording rules."""
import json
import re
from pathlib import Path

from .io import load_categories
from .evidence import _segments, is_heading, license_nonoperative_reason, audit_permission_present

PROFILE_PATH = Path(__file__).parent / 'config/category_profiles.json'
RULE_VERSION = '0.12-provisional'
SIGNALS = ('severity', 'scope', 'imbalance', 'exposure')
INFORMATIONAL = {'Agreement Date', 'Document Name', 'Effective Date', 'Expiration Date', 'Governing Law', 'Parties'}
CONTEXT_ONLY_RULES = {'audit_frequency_limit', 'assignment_void_consequence', 'audit_own_expense'}

# Each pair describes explicit broad and narrow scope for that category.
SCOPE_CUES = {
    'Affiliate License-Licensee': (r'all current and future affiliates', r'only the named affiliates'),
    'Affiliate License-Licensor': (r'rights of all current and future affiliates', r'rights of only the named affiliates'),
    'Anti-Assignment': (r'all direct and indirect transfers', r'only transfers of this agreement'),
    'Audit Rights': (r'audit all books and records', r'audit only records relating to this agreement'),
    'Cap On Liability': (r'cap applies to all claims', r'cap applies only to payment claims'),
    'Change Of Control': (r'any direct or indirect change of control', r'only a direct change of control'),
    'Covenant Not To Sue': (r'covenant covers all claims', r'covenant covers only the identified claims'),
    'Exclusivity': (r'exclusivity covers all products worldwide', r'exclusivity covers only the listed products'),
    'Insurance': (r'insurance covers all operations worldwide', r'insurance covers only services under this agreement'),
    'Ip Ownership Assignment': (r'assignment includes all background and future intellectual property', r'assignment covers only the listed deliverables'),
    'Irrevocable Or Perpetual License': (r'license continues in perpetuity', r'license applies only to the named project'),
    'Joint Ip Ownership': (r'joint ownership covers all background and future intellectual property', r'joint ownership covers only the listed deliverables'),
    'License Grant': (r'license permits all uses worldwide', r'license permits only internal use'),
    'Liquidated Damages': (r'liquidated damages apply to every breach', r'liquidated damages apply only to late delivery'),
    'Minimum Commitment': (r'minimum commitment applies to all products', r'minimum commitment applies only to the listed products'),
    'Most Favored Nation': (r'parity applies to all customers worldwide', r'parity applies only to equivalent transactions'),
    'No-Solicit Of Customers': (r'restriction covers all customers and prospects', r'restriction covers only the named customers'),
    'No-Solicit Of Employees': (r'restriction covers all employees worldwide', r'restriction covers only the named employees'),
    'Non-Compete': (r'restriction covers all business activities worldwide', r'restriction covers only the named activity'),
    'Non-Disparagement': (r'restriction covers all statements about either party', r'restriction covers only public statements about the project'),
    'Non-Transferable License': (r'transfer restriction covers all direct and indirect transfers', r'transfer restriction covers only the named license'),
    'Notice Period To Terminate Renewal': (r'notice requirement applies to all renewals', r'notice requirement applies only to the first renewal'),
    'Post-Termination Services': (r'transition services continue indefinitely', r'transition services cover only the listed tasks'),
    'Price Restrictions': (r'price restriction applies to all products worldwide', r'price restriction applies only to the listed products'),
    'Renewal Term': (r'renewal continues automatically indefinitely', r'renewal is limited to a single additional term'),
    'Revenue/Profit Sharing': (r'revenue sharing covers all business revenue', r'revenue sharing covers only revenue from the named product'),
    'Rofr/Rofo/Rofn': (r'right of first refusal covers all asset sales', r'right of first refusal covers only the named asset'),
    'Source Code Escrow': (r'escrow includes all source code and development tools', r'escrow includes only source code for the named product'),
    'Termination For Convenience': (r'termination right applies to all services and orders', r'termination right applies only to the named service'),
    'Third Party Beneficiary': (r'beneficiary rights extend to all affiliates and successors', r'beneficiary rights extend only to the named entity'),
    'Uncapped Liability': (r'unlimited liability applies to all claims', r'unlimited liability applies only to the identified claims'),
    'Unlimited/All-You-Can-Eat-License': (r'license permits unlimited users and all uses worldwide', r'license permits only internal use'),
    'Volume Restriction': (r'volume restriction covers all products and users', r'volume restriction covers only the named product'),
    'Warranty Duration': (r'warranty continues indefinitely', r'warranty covers only the listed deliverables'),
}

# Rules set one signal; repeated wording never stacks points.
RULES = [
    ('audit_unbounded_cost', ('Audit Rights',), 'exposure', r'audit costs are unlimited', 2, 'Audit costs are explicitly unbounded.'),
    ('audit_fixed_cost', ('Audit Rights',), 'exposure', r'audit costs are limited to a fixed fee', 1, 'A fixed fee bounds costs; its materiality remains unknown.'),
    ('audit_unilateral', ('Audit Rights',), 'imbalance', r'one party alone may expand the audit scope', 2, 'One party controls expansion of the audit.'),
    ('audit_reciprocal', ('Audit Rights',), 'imbalance', r'each party has identical audit rights and obligations', 0, 'The passage expressly describes reciprocal audit terms.'),
    ('audit_major_consequence', ('Audit Rights',), 'severity', r'audit failure triggers termination of all services', 2, 'The stated consequence affects all services.'),
    ('audit_limited_consequence', ('Audit Rights',), 'severity', r'the sole consequence of audit failure is a written reminder', 0, 'The stated sole consequence is limited.'),
    ('audit_relevant_records', ('Audit Rights',), 'scope',
     r'(?:inspect|audit|access)[^.!?;]{0,30}?(?:records|invoices) (?:relevant to|relating to|limited to)[^.!?;]{0,40}?(?:invoices|this agreement|the services)',
     0, 'Access is tied to records relevant to identified invoices or obligations.'),
    ('audit_all_financial_records', ('Audit Rights',), 'scope',
     r'(?:inspect|audit|access) (?:all|every) (?:financial|accounting) records?',
     2, 'Access expressly covers all financial or accounting records.'),
    ('audit_mutual_permission', ('Audit Rights',), 'imbalance',
     r'(?:either|each) party (?:may|has the right to) (?:inspect|audit|access)[^.!?;]{0,50}?records',
     0, 'The stated audit permission is available to both parties; other terms remain unverified.'),
    ('audit_exclusive_permission', ('Audit Rights',), 'imbalance',
     r'only (?:the )?(?:supplier|customer|buyer|seller|licensor|licensee|one party) (?:may|has the right to) (?:inspect|audit)[^.!?;]{0,50}?records',
     2, 'The text expressly reserves audit permission to one party.'),
    ('audit_shifted_costs', ('Audit Rights',), 'imbalance',
     r'(?:charge|shift) all (?:audit|inspection) (?:costs|expenses) to (?:the )?(?:supplier|customer|buyer|seller|licensor|licensee|other party)',
     1, 'The inspecting side shifts all stated audit costs to the counterparty; amount and overall reciprocity remain unknown.'),
    ('audit_own_expense', ('Audit Rights',), 'exposure',
     r'(?:inspect|audit)[^.!?;]{0,180}?at (?:its|their|the auditing party\x27s) own expense',
     1, 'The inspecting party bears its costs; this does not establish their magnitude or a cap.'),
    ('audit_service_consequence', ('Audit Rights',), 'severity',
     r'(?:inspect|audit)[^.!?;]{0,220}?(?:suspend|terminate) (?:all )?services for (?:refusal|failure to cooperate)',
     2, 'Refusing the audit expressly permits service suspension or termination.'),
    ('liability_unbounded', ('Cap On Liability', 'Uncapped Liability'), 'exposure', r'liability is (?:unlimited|uncapped)', 2, 'Liability is expressly unbounded.'),
    ('liability_fixed_cap', ('Cap On Liability', 'Uncapped Liability'), 'exposure', r'aggregate liability is capped at a fixed amount', 1, 'A stated cap bounds exposure; adequacy is unresolved.'),
    ('damages_unbounded', ('Liquidated Damages',), 'exposure', r'liquidated damages accrue without limit', 2, 'Accrual is expressly unbounded.'),
    ('damages_bounded', ('Liquidated Damages',), 'exposure', r'liquidated damages are capped at a fixed amount', 1, 'A fixed cap bounds accrual.'),
    ('transition_unbounded', ('Post-Termination Services',), 'exposure', r'transition service costs are unlimited', 2, 'Transition costs are expressly unbounded.'),
    ('exit_unilateral', ('Termination For Convenience',), 'imbalance', r'one party alone may terminate for convenience', 2, 'Only one party has the stated discretionary exit right.'),
    ('exit_reciprocal', ('Termination For Convenience',), 'imbalance', r'each party has identical termination rights', 0, 'The stated exit rights are reciprocal.'),
    ('exit_either_party', ('Termination For Convenience',), 'imbalance',
     r'(?:either|each) party may terminate (?:this|the) agreement (?:without cause|for convenience)'
     r'(?: upon (?:[a-z-]+|\d+)(?: \(\d+\))? days[\x27\u2019]? (?:prior )?written notice to the other party)?(?=\s*[.!?]|\s*$)',
     0, 'Both parties receive this same stated termination right; notice length does not establish severity or exposure.'),
    ('license_purpose_restricted', ('License Grant',), 'scope',
     r'solely for this purpose, (?:(?!(?:to|shall|must|may|will|can|pays?|maintains?|agrees?|requests?)\b)[\w&\x27-]+ ){1,6}grants? '
     r'(?:[\w\x27-]+ ){1,3}a (?:non-exclusive, |royalty-free, |limited )*'
     r'(?:right|license) to use (?:the )?[\w-]+(?=\s*[.!?]|\s*$)',
     0, 'The purpose restriction directly qualifies the express grant of use rights; other signals remain unverified.'),
    ('exit_major_consequence', ('Termination For Convenience',), 'severity', r'termination immediately ends all operational services', 2, 'The stated consequence ends all operational services.'),
    ('license_worldwide', ('License Grant', 'Affiliate License-Licensee', 'Affiliate License-Licensor'),
     'scope', r'grant(?:s|ed)?[^.!?;]{0,160}?world[- ]?wide[^.!?;]{0,100}?(?:licen[cs]e|right)',
     2, 'The grant expressly has worldwide geographic scope; purpose and use restrictions still require review.'),
    ('license_use_restricted', ('License Grant',), 'scope',
     r'grants?[^.!?;]{0,140}?\b(?:license|right) to use '
     r'(?:(?!\b(?:pay\w*|maintain\w*|report\w*|fees|shall|must)\b)[^.!?;]){1,180}?'
     r'(?:solely for (?:such )?[\w -]+ purposes|solely in connection with (?:the )?[\w -]+|only in combination with (?:the )?[\w -]+)',
     0, 'The stated use restriction directly limits the granted license; other risk dimensions remain unverified.'),
    ('license_perpetual', ('License Grant', 'Irrevocable Or Perpetual License'),
     'scope', r'(?:perpetual[^.!?;]{0,80}?licen[cs]e|licen[cs]e[^.!?;]{0,80}?in perpetuity)',
     2, 'The license expressly has perpetual duration; other dimensions remain subject to its terms.'),
    ('audit_frequency_limit', ('Audit Rights',), 'scope',
     r'(?:once every (?:twelve(?: \(12\))? months|year)|not more than once (?:per|in (?:any|each)) (?:calendar )?year)[^.!?;]{0,100}?(?:inspect|audit)',
     1, 'Annual frequency is recorded, but does not establish overall scope.'),
    ('liability_stated_cap', ('Cap On Liability', 'Uncapped Liability'), 'exposure',
     r'(?:aggregate |total )?liability[^.!?;]{0,160}?(?:shall not exceed|shall be limited to|is limited to)[^.!?;]{0,160}?(?:dollars|amount[^.!?;]{0,80}?paid|fees[^.!?;]{0,80}?paid)',
     1, 'An explicit monetary or paid-fee cap bounds this liability; cap adequacy is not assessed.'),
    ('assignment_void_consequence', ('Anti-Assignment',), 'severity',
     r'(?:such |purported |any )?assignment[^.!?;]{0,120}?(?:shall be|is) (?:null and )?void',
     1, 'The stated assignment consequence is recorded; business severity is not established.'),
    ('assignment_consent_discretion', ('Anti-Assignment',), 'imbalance',
     r'consent[^.!?;]{0,100}?(?:withheld|granted)[^.!?;]{0,60}?sole discretion',
     1, 'Consent is subject to discretionary control; this does not establish overall unfairness or exclusive rights.'),
    ('ip_defined_deliverables', ('Ip Ownership Assignment',), 'scope',
     r'(?:assign(?:s|ment)?|ownership)[^.!?;]{0,80}?(?:only|solely)[^.!?;]{0,60}?(?:deliverables|work product)',
     0, 'The transfer is expressly restricted to identified work output; other rights require context.'),
    ('ip_background_future', ('Ip Ownership Assignment',), 'scope',
     r'(?:assign(?:s|ment)?|ownership)[^.!?;]{0,100}?all[^.!?;]{0,60}?background and future (?:intellectual property|inventions)',
     2, 'The transfer expressly includes both background and future rights.'),
    ('insurance_project_scope', ('Insurance',), 'scope',
     r'insurance[^.!?;]{0,80}?(?:covers only|limited to)[^.!?;]{0,60}?(?:project|services|operations)',
     0, 'The insurance obligation expressly covers a limited activity.'),
    ('insurance_reciprocal', ('Insurance',), 'imbalance',
     r'each party[^.!?;]{0,60}?(?:shall|agrees to)[^.!?;]{0,60}?(?:carry|maintain)[^.!?;]{0,40}?insurance',
     0, 'Both parties have the stated coverage obligation; adequacy and other terms remain unverified.'),
    ('minimum_capped_payments', ('Minimum Commitment',), 'exposure',
     r'(?:minimum commitment|minimum purchase|minimum payment)[^.!?;]{0,80}?(?:capped at|shall not exceed)[^.!?;]{0,50}?(?:dollars|\$[0-9,]+)',
     1, 'An explicit limit bounds the commitment; financial materiality is unknown.'),
    ('minimum_all_products', ('Minimum Commitment',), 'scope',
     r'(?:minimum purchase|minimum payment|minimum commitment)[^.!?;]{0,60}?(?:all products|all services)',
     2, 'The minimum applies across all stated products or services.'),
    ('sharing_named_product', ('Revenue/Profit Sharing',), 'scope',
     r'(?:royalt(?:y|ies)|revenue sharing|profit sharing)[^.!?;]{0,80}?(?:only|solely)[^.!?;]{0,60}?(?:named product|licensed product)',
     0, 'Sharing is expressly restricted to the identified product revenue.'),
    ('sharing_unilateral_formula', ('Revenue/Profit Sharing',), 'imbalance',
     r'one party alone may (?:change|determine)[^.!?;]{0,60}?(?:royalty|revenue sharing|profit sharing)[^.!?;]{0,30}?(?:rate|formula)',
     2, 'The text expressly gives one party exclusive control of the sharing formula.'),
]

# Complex qualifications are deferred rather than interpreted by keyword alone.
UNCERTAINTY = re.compile(
    r'\b(?:not|no|never|neither|except\w*|unless|notwithstanding|provided that|'
    r'subject to|however|if|might|could|section|schedule|exhibit|whether|discuss(?:ed|ion|ing)?|hypothetical|suppose|assuming|proposed)\b', re.I)


# A matched positive limitation can itself contain "not" (e.g. "shall not exceed").
KNOWN_LIMITS = re.compile(r'\b(?:shall not exceed|not more than once)\b', re.I)


def rule_context(text, start, end):
    segments = list(_segments(text))
    indices = [index for index, (left, right) in enumerate(segments) if left < end and right > start]
    if not indices:
        return text
    first, last = min(indices), max(indices)
    left, right = segments[first][0], segments[last][1]
    context = text[left:right]
    if first:
        previous = text[segments[first - 1][0]:left]
        if re.search(r'\b(?:except|unless|notwithstanding|provided that|subject to)\b', previous, re.I):
            context = previous + context
    following = text[right:]
    if re.match(r'[\s.!?]*(?:except|unless|however|notwithstanding|provided that|subject to|this (?:cap|limit|limitation|restriction)|these (?:limits|limitations|restrictions))\b', following, re.I):
        context += following
    return context


def match_is_uncertain(text, start, end):
    context = rule_context(text, start, end)
    context = KNOWN_LIMITS.sub('stated limit', context)
    return bool(UNCERTAINTY.search(context)) or '?' in context


def load_profiles(path=PROFILE_PATH):
    data = json.loads(Path(path).read_text())
    profiles = data['profiles']
    if set(profiles) != set(load_categories()):
        raise ValueError('Profiles must match all 41 categories')
    for profile in profiles.values():
        scores = profile['initial_scores']
        if set(scores) != set(SIGNALS) or any(type(score) is not int or not 0 <= score <= 2 for score in scores.values()):
            raise ValueError('Profile scores must be integers from 0 to 2')
    return data


def risk_level(total):
    if type(total) is not int or not 0 <= total <= 8:
        raise ValueError('Total must be an integer from 0 to 8')
    return 'Low' if total <= 2 else 'Medium' if total <= 5 else 'High'


def category_evidence_supported(text, category):
    """Relevance only, not interpretation or a new severity rule.

    Ignore headings; require operative transfer terms or liability/remedy
    limitations in the same sentence. Existing uncertainty checks still apply.
    Other categories retain their existing eligibility policy.
    """
    if category == 'Audit Rights':
        return audit_permission_present(text)
    if category == 'Exclusivity':
        # Only reject the two known non-operative forms, and retain passages
        # containing a separate positive commitment (including mixed periods).
        nonexclusive_pattern = r'\bnon(?:\s*-\s*|\s+)exclusive\b'
        # Mask only non-exclusive wording while checking for a separate
        # positive commitment. Preserve original evidence and offsets.
        commitment_text = re.sub(nonexclusive_pattern, lambda match: ' ' * len(match.group()), text, flags=re.I)
        exclusive = r'\bexclusiv(?:e|ely)\b'
        commitment = re.search(
            r'\b(?:shall|will|must|is|are|agrees?\s+to)\b[^.!?;]{0,180}' + exclusive + r'|'
            r'\b(?:grants?|appoints?)\b[^.!?;]{0,100}' + exclusive + r'|'
            r'\bpurchases?\b[^.!?;]{0,100}\bexclusively\b', commitment_text, re.I)
        nonexclusive = re.search(nonexclusive_pattern, text, re.I)
        definition = re.search(r'\b(?:means|defined\s+as|deemed)\b', text, re.I)
        return bool(commitment) or not (nonexclusive or definition)
    if category not in {'Anti-Assignment', 'Cap On Liability'}:
        return True
    for left, right in _segments(text):
        sentence = text[left:right].strip()
        if is_heading(sentence, category):
            continue
        if category == 'Anti-Assignment':
            transfer = re.search(r'\b(?:assign(?:ment|ed|ee|ees)?|transfer(?:red|able|s)?)\b', sentence, re.I)
            terms = re.search(
                r'\b(?:consent\w*|approv\w*|prohibit\w*|restrict\w*|void|notify|notification|'
                r'non[- ]assignable|non[- ]transferable|may\s+(?:not\s+)?(?:assign|transfer)|'
                r'(?:assignments?|transfers?)\s+(?:is|are|shall\s+be)\s+(?:permitted|allowed)|'
                r'fully\s+transferable|entitled\s+to\s+(?:assign|transfer)|'
                r'remain\s+(?:liable|bound)|agree\s+in\s+writing\s+to\s+be\s+bound)\b', sentence, re.I)
            if transfer and terms:
                return True
        else:
            if re.search(
                    r'\b(?:liabilit(?:y|ies)|damages)\b(?!\s+insurance\b)[^.!?;]{0,240}\b(?:limited\s+to|capped\s+at|shall\s+not\s+exceed)\b|'
                    r'\bremedies\s+shall\s+be\s+limited\s+to\b|'
                    r'\bin\s+no\s+event\b[^.!?;]{0,240}\b(?:liability|liable)\b[^.!?;]{0,160}\b(?:exceed|damages)\b|'
                    r'\b(?:sole|exclusive)\s+(?:(?:obligation|liability)\s+and\s+)?remed(?:y|ies)\b[^.!?;]{0,240}\b(?:is|are|shall|will)\b|'
                    r'\b(?:entire|sole)\s+liability\b[^.!?;]{0,240}\b(?:is|shall|will)\b|'
                    r'\b(?:states|constitutes)\s+the\s+entire\s+liability\b|'
                    r'\blimitation\s+of\s+liability\b[^.!?;]{0,240}\b(?:shall|is|applies)\b|'
                    r'\b(?:neither\s+party|\w+\s+shall\s+not)\s+(?:be\s+)?liable\b',
                    sentence, re.I):
                return True
    return False


def score_evidence(evidence, profiles=None):
    """Score one record returned by extract_evidence, preserving its provenance."""
    profiles = load_profiles() if profiles is None else profiles
    category = evidence['category']
    profile = profiles['profiles'][category]
    text = evidence['original_text']
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Evidence text must not be empty')
    start = evidence['char_start']
    if type(start) is not int or start < 0 or evidence['char_end'] - start != len(text):
        raise ValueError('Evidence offsets do not match text')
    scores = profile['initial_scores'].copy()
    reasons = set(evidence.get('review_reasons', []))
    limitations = set(evidence.get('limitations', []))
    ordinary = {'heuristic_boundaries', 'provisional_rubric_and_wording_rules', 'no_wording_rule_matched'}
    limitations.update(reasons & ordinary)
    reasons -= ordinary
    limitations.add('provisional_rubric_and_wording_rules')
    rules = [rule for rule in RULES if category in rule[1]]
    if category in SCOPE_CUES:
        broad, narrow = SCOPE_CUES[category]
        rules += [
            ('broad_scope', (category,), 'scope', broad, 2, 'Explicit broad coverage.'),
            ('narrow_scope', (category,), 'scope', narrow, 0, 'Explicit restricted coverage.'),
        ]
    uncertain = bool(UNCERTAINTY.search(KNOWN_LIMITS.sub('stated limit', text))) or '?' in text
    matches = []
    for rule_id, _, signal, pattern, value, explanation in rules:
        for match in re.finditer(r'\b(?:' + pattern.replace(' ', r'\s+') + r')\b', text, re.I):
            matches.append({
                'rule_id': category + ':' + rule_id, 'signal': signal,
                'proposed_score': value, 'quote': match.group(),
                'char_start': start + match.start(), 'char_end': start + match.end(),
                'explanation': explanation,
                'context_uncertain': match_is_uncertain(text, match.start(), match.end()) or (
                    rule_id in {'license_purpose_restricted', 'license_use_restricted', 'exit_either_party'} and uncertain) or (
                    rule_id == 'audit_all_financial_records' and bool(re.search(
                        r'\b(?:only|solely|limited to|relevant to|relating to)\b', rule_context(text, match.start(), match.end()), re.I))) or (
                    rule_id in {'license_worldwide', 'license_perpetual'} and bool(re.search(
                        r'\b(?:only|solely|soley|limited to|subject always|for the duration|during the duration|prior\s+written\s+consent|for\s+the\s+term\s+of|with\s+respect\s+to\s+and\s+in\s+conjunction\s+with)\b', text, re.I))),
                'context_only': rule_id in CONTEXT_ONLY_RULES,
            })
    if category == 'License Grant' and re.search(r'\bworld[- ]?wide\b', text, re.I) and any(
            m['rule_id'].endswith((':license_purpose_restricted', ':license_use_restricted')) for m in matches):
        reasons.add('license_scope_dimensions_conflict')
        for match in matches:
            if match['signal'] == 'scope':
                match['context_uncertain'] = True
    unlocalized = evidence.get('localization') != 'heuristic_passage' or evidence.get('usable_for_scoring') is False
    blocked = unlocalized or (category == 'License Grant' and bool(license_nonoperative_reason(text)))
    if uncertain:
        reasons.add('negation_condition_or_reference_unresolved')
    if unlocalized:
        reasons.add('wording_adjustments_deferred_for_unlocalized_text')
    signal_sources = dict.fromkeys(SIGNALS, 'profile_default_unverified')
    for signal in SIGNALS:
        signal_matches = [match for match in matches if match['signal'] == signal]
        eligible = [match for match in signal_matches if not match['context_uncertain'] and not match['context_only']]
        values = {match['proposed_score'] for match in eligible}
        conflict = len({match['proposed_score'] for match in signal_matches if not match['context_only']}) > 1
        if conflict:
            reasons.add('conflicting_rules:' + signal)
        apply = bool(values) and not blocked and not conflict
        if apply:
            scores[signal] = next(iter(values))
            signal_sources[signal] = 'wording_rule_provisional'
        for match in signal_matches:
            match['status'] = ('deferred_uncertainty' if blocked or match['context_uncertain']
                               else 'observed_context_only' if match['context_only']
                               else 'applied' if apply else 'deferred_conflict')
            match['starting_score'] = profile['initial_scores'][signal]
            match['final_score'] = scores[signal]
    if not matches:
        limitations.add('no_wording_rule_matched')
    if any(match['context_only'] for match in matches):
        reasons.add('context_does_not_establish_signal')
    if any(match['context_uncertain'] for match in matches):
        reasons.add('wording_adjustment_requires_context')
    substantive = category not in INFORMATIONAL and category != 'Competitive Restriction Exception'
    if substantive and signal_sources['severity'] == 'profile_default_unverified':
        limitations.add('business_consequence_unverified')
    if substantive and signal_sources['imbalance'] == 'profile_default_unverified':
        limitations.add('party_balance_unknown_not_confirmed')
    financial_categories = {'Insurance', 'Minimum Commitment', 'Revenue/Profit Sharing',
                            'Liquidated Damages', 'Cap On Liability', 'Uncapped Liability'}
    if category in financial_categories:
        if signal_sources['exposure'] == 'profile_default_unverified':
            reasons.add('financial_exposure_unverified')
        elif scores['exposure'] == 1:
            reasons.add('financial_materiality_unknown')
    eligible_for_triage = evidence.get('usable_for_scoring', not blocked) and not uncertain and not any(match['context_uncertain'] for match in matches)
    if not eligible_for_triage:
        reasons.add('numeric_triage_deferred_for_uncertainty')
    if category == 'Uncapped Liability' and not any(
            match['rule_id'].endswith(':liability_unbounded') and match['status'] == 'applied'
            for match in matches):
        eligible_for_triage = False
        reasons.add('unlimited_liability_not_established')
    if not category_evidence_supported(text, category):
        eligible_for_triage = False
        substantive = False
        reasons.add('category_relevant_provision_not_established')
    if category == 'License Grant' and (reason := license_nonoperative_reason(text)):
        eligible_for_triage = False
        substantive = False
        reasons.add(reason)
    insurance_matches = list(re.finditer(
            r'\b(?:shall|must|agrees? to|will|required to)\s+(?:at (?:its|their) (?:own )?expense,?\s+)?(?:maintain|carry|obtain|procure|provide|purchase)\b[^.!?;]{0,100}\binsurance\b', text, re.I))
    if category == 'Insurance' and not any(not match_is_uncertain(text, match.start(), match.end()) for match in insurance_matches):
        eligible_for_triage = False
        reasons.add('substantive_insurance_obligation_not_established')
    if category == 'License Grant' and re.search(
            r'\b(?:fully\s+licensed|licensed\s+and\s+registered|licensing\s+status)\b', text, re.I):
        grants = list(re.finditer(r'\bgrants?\b[^.!?;]{0,160}\b(?:rights?|licen[cs]e)\b', text, re.I))
        if not any(not match_is_uncertain(text, match.start(), match.end()) for match in grants):
            eligible_for_triage = False
            substantive = False
            reasons.add('licensing_status_is_not_a_rights_grant')
    total = sum(scores.values())
    return {
        **evidence,
        'starting_scores': profile['initial_scores'].copy(),
        'substantive_evidence': substantive, 'usable_for_scoring': eligible_for_triage,
        'numeric_score_status': 'provisional' if eligible_for_triage else 'deferred_review',
        'risk_scores': scores, 'risk_total': total, 'risk_level': risk_level(total),
        'risk_status': 'provisional_heuristic_not_human_reviewed',
        'signal_sources': signal_sources, 'rule_matches': matches,
        'profile_version': profiles['profile_version'], 'rule_version': RULE_VERSION,
        'needs_review': bool(reasons), 'review_reasons': sorted(reasons),
        'limitations': sorted(limitations),
    }


def score_clauses(evidence_records):
    profiles = load_profiles()
    return [score_evidence(record, profiles) for record in evidence_records]
