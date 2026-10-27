"""PDTB sense labels: normalisation and three-level projection.

The evaluation reports accuracy/kappa at three granularities:

    L3  full sense string, e.g. ``CONTINGENCY:Cause:Reason``
    L2  first two segments, e.g. ``CONTINGENCY:Cause``
    L1  top-level class,   e.g. ``CONTINGENCY``

This matters because published agreement figures are not comparable across
granularities: Chinese-DiMLex reports Fleiss kappa=0.539 on fine-grained
senses, while Zhu & Xu (LREC 2026) report kappa=0.941 on 13 coarse UD
relation types.  Any comparison has to state which level it is on.
"""

from __future__ import annotations

L1_CLASSES = ("TEMPORAL", "CONTINGENCY", "COMPARISON", "EXPANSION")

L1_ZH = {
    "TEMPORAL": "时间",
    "CONTINGENCY": "因果条件",
    "COMPARISON": "对比",
    "EXPANSION": "扩展",
}

#: Common surface variants that appear across PDTB 2.0 / 3.0 / Chinese treebanks.
_ALIASES = {
    "CAUSE": "CONTINGENCY:Cause",
    "RESULT": "CONTINGENCY:Cause:Result",
    "REASON": "CONTINGENCY:Cause:Reason",
    "CONDITION": "CONTINGENCY:Condition",
    "CONTRAST": "COMPARISON:Contrast",
    "CONCESSION": "COMPARISON:Concession",
    "CONJUNCTION": "EXPANSION:Conjunction",
    "TEMPORAL:ASYNCHRONOUS": "TEMPORAL:Asynchronous",
    "TEMPORAL:SYNCHRONOUS": "TEMPORAL:Synchronous",
}

_LEVEL2_ALIASES = {
    # PDTB 2.0 names that were renamed in 3.0
    "CONTINGENCY:PRAGMATIC_CAUSE": "CONTINGENCY:Pragmatic cause",
    "CONTINGENCY:PRAGMATIC CONDITION": "CONTINGENCY:Pragmatic condition",
}


def normalize_sense(raw: str | None) -> str:
    """Canonicalise a sense string without changing its granularity."""
    if not raw:
        return ""
    s = " ".join(raw.strip().split())
    # PDTB 3.0 files (and TED-CDB) use dots as the level separator:
    # "Contingency.Cause.Reason".  CDTB/DiMLex use colons.  Canonicalise to ":".
    s = s.replace(".", ":").replace("_", " ").replace("/", ":").replace("  ", " ")
    # "Contingency:Cause:reason" -> "CONTINGENCY:Cause:reason"
    parts = [p.strip() for p in s.split(":") if p.strip()]
    if not parts:
        return ""
    parts[0] = parts[0].upper()
    # Segment casing is inconsistent across PDTB releases and DiMLex
    # ("Negative-Condition" vs "Negative-condition").  Canonical form:
    # first letter upper, remainder lower.  "Arg1-as-denier" is unaffected.
    parts[1:] = [p[:1].upper() + p[1:].lower() for p in parts[1:]]
    s = ":".join(parts)
    s = _ALIASES.get(s, s)
    return _LEVEL2_ALIASES.get(s.upper(), s)


def sense_levels(raw: str | None) -> tuple[str, str, str]:
    """Return ``(L1, L2, L3)`` for a sense string.

    ``L3`` keeps the full normalised sense; ``L1``/``L2`` are prefixes.
    Unknown or empty senses yield empty strings at every level.
    """
    s = normalize_sense(raw)
    if not s:
        return "", "", ""
    parts = s.split(":")
    l1 = parts[0].upper()
    l2 = ":".join([l1] + parts[1:2]) if len(parts) > 1 else l1
    return l1, l2, s


def is_known_level1(sense: str | None) -> bool:
    return (sense or "").split(":")[0].upper() in L1_CLASSES
