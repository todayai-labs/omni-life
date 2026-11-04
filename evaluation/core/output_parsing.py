"""Parse explicit model output without guessing actions or changing raw text."""
import re
import unicodedata

CONTROL = re.compile(r'<(?:unk|\|(?:eot_id|round_end|im_end|endoftext|end_of_text|begin_of_text|im_start|start_header_id|end_header_id)\|)>')
TAG = re.compile(r'^\s*\[([A-Za-z_-]+)\]\s*')
ACTIONS = {'warn', 'inform', 'feedback', 'answer', 'wait', 'count', 'summarize'}

def clean_native_output(raw):
    """Remove only allowlisted native control tokens, never arbitrary <...>."""
    text = unicodedata.normalize('NFKC', raw if isinstance(raw, str) else '')
    text = CONTROL.sub('', text).strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:[A-Za-z0-9_-]+)?\s*\n?', '', text)
        text = text[:-3].strip()
    return text

def parse_action(raw, action=None):
    text = clean_native_output(raw)
    # Auxiliary bracket prefixes may precede the explicit behavior tag.
    # Accept only a contiguous bracket-only prefix, never search through prose.
    prefix = re.match(r'^(?:\s*\[[A-Za-z_-]+\])+\s*', text)
    if prefix:
        tags = list(re.finditer(r'\[([A-Za-z_-]+)\]', prefix[0]))
        known = [m for m in tags if m[1].lower() in ACTIONS]
        if len(known) == 1:
            text = text[known[0].start():]
    match = TAG.match(text)
    tag = match[1].lower() if match else None
    body = text[match.end():].strip() if match else text
    field = action.strip().lower().strip('[]') if isinstance(action, str) else None
    field = field or None
    reason = None
    if field and field not in ACTIONS: reason = 'unknown_action_field'
    elif tag and tag not in ACTIONS: reason = 'unknown_action_tag'
    elif tag and field and tag != field: reason = 'conflicting_action_field_and_tag'
    elif re.search(r'\[(?:warn|inform|feedback|answer|wait|count|summarize)\]', body, re.I): reason = 'multiple_action_tags'
    elif not (field or tag): reason = 'missing_explicit_action'
    elif (field or tag) == 'wait' and body: reason = 'wait_has_spoken_content'
    return {'action': None if reason else field or tag, 'content': body, 'valid': reason is None, 'reason': reason}

def parse_choices(raw, allowed, options=None):
    text = re.sub(r'<think>.*?</think>', '', clean_native_output(raw), flags=re.S).strip().upper()
    text = re.sub(r'^(?:答案|ANSWER)\s*[:：]\s*', '', text)
    if not re.fullmatch(r'[A-L](?:\s*[;,，；、\s]\s*[A-L])*[。.]?', text):
        match = re.fullmatch(r'([A-L])[.．、：:]\s*(.+)', text, flags=re.S)
        norm = lambda s: re.sub(r'\s+', '', clean_native_output(s)).rstrip('。.').upper()
        if match and options and match[1] in options and norm(match[2]) == norm(options[match[1]]): return [match[1]], True
        return [], False
    picks = sorted(set(re.findall('[A-L]', text)))
    return picks, bool(picks and set(picks) <= set(allowed))
