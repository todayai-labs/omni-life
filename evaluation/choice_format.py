"""Official QA choice parsing functions for the frozen results."""
import re
import unicodedata
SELECTION = re.compile(r"[A-L](?:[ \t\r\n,;、]+[A-L])*")
CONTROL = re.compile(r"<(?:unk|\|(?:eot_id|round_end|im_end|endoftext|end_of_text|begin_of_text|im_start|start_header_id|end_header_id)\|)>")


def parse_selected(raw, mode, allowed):
    """Parse the submitted selection field. Invalid or blank selections earn zero."""
    if mode not in ("single", "multiple"):
        raise ValueError("Unknown QA response mode")
    if not isinstance(raw, str):
        return ()
    text = unicodedata.normalize("NFKC", raw).strip().upper()
    if not SELECTION.fullmatch(text):
        return ()
    picks = re.findall("[A-L]", text)
    if (mode == "single" and len(picks) != 1) or not set(picks).issubset(allowed):
        return ()
    return tuple(sorted(set(picks)))


def parse_model_output(raw, mode, options):
    """Read one explicit final selection, preserving multiplicity until validation."""
    if not isinstance(raw, str):
        return ()
    text = CONTROL.sub("", unicodedata.normalize("NFKC", raw)).strip()
    text = re.sub(r"^<think>.*?</think>\s*", "", text, count=1, flags=re.S)
    fenced = re.fullmatch(r"```(?:[\w-]+)?[ \t]*\r?\n(.*?)\r?\n?```", text, flags=re.S)
    if fenced:
        text = fenced[1].strip()
    text = re.sub(r"^(?:答案|ANSWER)\s*:\s*", "", text, flags=re.I).strip()
    # A verbatim option echo still expresses exactly one choice.
    echo = re.fullmatch(r"([A-L])[.、:]\s*(.+)", text, flags=re.I | re.S)
    if echo and echo[1].upper() in options:
        def normalized(value):
            return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).rstrip("。.").casefold()
        label = echo[1].upper()
        if normalized(echo[2]) == normalized(options[label]):
            return (label,)
    return parse_selected(text.removesuffix("。").removesuffix("."), mode, options)
