"""Prompt templates and output parsing.

Three regimes are fixed in the experiment plan:

    zero_shot   definitions only
    few_shot    definitions + k stratified exemplars
    cot         few-shot + "reason first, then JSON" (appendix only)

Every template ends with a strict JSON contract so predictions are machine
parseable.  Parsing is defensive: models do add prose around JSON.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Sequence

from .labels import L1_ZH
from .schema import Item

#: Bump whenever a template changes.  Predictions carry this string, so a run
#: from an older prompt can always be told apart from a run on the current one.
PROMPT_VERSION = "l1-1"

DEFINITIONS = """PDTB 3.0 篇章关系义项体系（四个大类）：
- TEMPORAL（时间）：两个情境在时间上相关，含 Synchronous（同时）、Asynchronous（先后：Precedence / Succession）。
- CONTINGENCY（因果条件）：一个情境导致、使能或阻止另一个情境，含 Cause（因果）、Condition（条件）、Pragmatic cause 等。
- COMPARISON（对比）：两个情境形成对照或让步，含 Contrast（对比）、Concession（让步）。
- EXPANSION（扩展）：第二个论元对第一个作扩展、说明或替代，含 Conjunction（并列）、Instantiation（举例）、Restatement（重述）、Alternative（选择）、Exception（例外）、Substitution（替换）。

标签格式为「大类:二级[:三级]」，例如 CONTINGENCY:Cause:Result。请尽量给出二级或三级义项。"""

#: Top-level-only prompt.  Asking for L1 while the template demands L2/L3 is a
#: different task: the model spends probability mass on distinctions that are
#: then discarded, and parsing failures rise.  Every L1 experiment uses this.
DEFINITIONS_L1 = """汉语篇章关系四大类（PDTB 3.0 顶层）：
- TEMPORAL（时间）：两个情境在时间上相关，含同时与先后。
- CONTINGENCY（因果条件）：一个情境导致、使能或阻止另一个情境，含因果、条件、目的、推断。
- COMPARISON（对比）：两个情境形成对照或让步。
- EXPANSION（扩展）：第二个论元扩展、说明、细化或替代第一个，含并列、递进、举例、例外、选择、列表。

只需判定大类。输出必须是下列四个英文标签之一：TEMPORAL / CONTINGENCY / COMPARISON / EXPANSION。"""

SYSTEM = "你是汉语篇章语言学专家，熟悉 PDTB 篇章关系标注体系。只输出 JSON，不要输出其他内容。"

#: Rewordings of the L1 instruction used only by R171.  They change *when* and
#: *how* the task is stated, not what counts as correct: definitions and the
#: output contract stay identical, so any movement in kappa is attributable to
#: wording rather than to a different task.
PROMPT_VARIANTS = {
    "v0": {},
    "v1": {
        "preface": (
            "任务：下面给出汉语篇章中的两个论元（Arg1 与 Arg2）以及它们之间的连接词。"
            "请判断 Arg1 与 Arg2 之间的篇章关系属于四大类中的哪一类。只判大类。"
        ),
    },
    "v2": {
        "rules": (
            "判断顺序：① 先看同时管辖 Arg1 与 Arg2 的连接词，它通常直接决定大类；"
            "② 没有跨论元连接词时，看 Arg2 对 Arg1 做了什么——推进时间线（TEMPORAL）、"
            "给出原因或结果（CONTINGENCY）、形成对照或让步（COMPARISON）、补充说明或举例（EXPANSION）；"
            "③ 只管辖论元内部成分的连接词不能决定 Arg1↔Arg2 的关系。"
        ),
    },
}


def _render_explicit(item: Item, level: str = "l3") -> str:
    task = "篇章关系大类" if level == "l1" else "篇章关系义项"
    variant = item.meta.get("variant")
    # "no_conn_tokens" is the paired-frame fix: only the frame's own tokens are
    # replaced, so it is the same instruction and the same rendering.
    if variant in {"no_connective", "no_conn_tokens"}:
        return (
            f"句子（原连接词已被删除，用 ____ 标出）：\n{item.text}\n\n"
            f"请判断此处原本表达的{task}。"
        )
    if item.conn_char_start >= 0 and item.conn_char_end > item.conn_char_start:
        s, e = item.conn_char_start, item.conn_char_end
        marked = f"{item.text[:s]}【{item.text[s:e]}】{item.text[e:]}"
    else:
        marked = item.text
    return (
        f"句子（连接词已用【】标出）：\n{marked}\n\n"
        f"请判断连接词「{item.connective}」在此语境中的{task}。"
    )


def _render_implicit(item: Item, level: str = "l3") -> str:
    task = "篇章关系大类" if level == "l1" else "篇章关系义项"
    return (
        "下面两个论元之间没有显性连接词。\n"
        f"论元 1：{item.arg1}\n论元 2：{item.arg2}\n\n"
        f"请判断二者之间的隐式{task}。"
    )


def _example_block(items: Sequence[Item], level: str = "l3") -> str:
    lines = []
    for it in items:
        if it.task == "explicit_sense" and it.connective:
            q = f"连接词「{it.connective}」；句子：{it.text}"
        else:
            q = f"论元1：{it.arg1}；论元2：{it.arg2}"
        answer = it.sense_l1 if level == "l1" else it.gold
        lines.append(f"示例\n{q}\n答案：{{\"sense\": \"{answer}\"}}")
    return "\n\n".join(lines)


SELF_ASSESS_FORMAT = (
    '输出格式（严格 JSON）：{"sense": "<TEMPORAL|CONTINGENCY|COMPARISON|EXPANSION>", '
    '"confidence": <0-100 的整数，表示你对这个判断的把握>}'
)


def build_self_assess_messages(item: Item) -> list[dict[str, str]]:
    """L1 prompt that also asks the model to rate its own certainty.

    The label is identical to the ordinary L1 task; only the output contract
    changes, so accuracy on this run is directly comparable to the plain run.
    """
    msgs = build_messages(item, regime="zero_shot", level="l1")
    marker = '输出格式（严格 JSON）：{"sense": "<TEMPORAL|CONTINGENCY|COMPARISON|EXPANSION>"}'
    user = msgs[1]["content"]
    if marker not in user:
        raise ValueError("L1 output contract changed; update build_self_assess_messages")
    return [msgs[0], {"role": "user", "content": user.replace(marker, SELF_ASSESS_FORMAT)}]


def parse_self_assess(raw: str, allowed: Sequence[str] | None = None) -> tuple[str, float | None]:
    """Return the label and the self-reported confidence rescaled to 0--1."""
    sense = parse_sense(raw, allowed)
    conf = None
    for blob in _JSON_OBJ.findall(raw or ""):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            for key in ("confidence", "把握", "置信度", "certainty"):
                val = obj.get(key)
                if isinstance(val, (int, float)):
                    conf = max(0.0, min(1.0, float(val) / 100.0))
                    break
                if isinstance(val, str):
                    try:
                        conf = max(0.0, min(1.0, float(val.strip().rstrip("%")) / 100.0))
                        break
                    except ValueError:
                        continue
        if conf is not None:
            break
    return sense, conf


def build_messages(
    item: Item,
    regime: str = "zero_shot",
    shots: Iterable[Item] = (),
    extra_senses: Sequence[str] = (),
    level: str = "l3",
    variant: str = "v0",
) -> list[dict[str, str]]:
    """Render one prompt.  ``level="l1"`` switches to the 4-class task.

    Asking for L1 while the template still demands L2/L3 is not the same task:
    probability mass goes to distinctions that are then thrown away, and the
    output contract is harder to parse.  L1 experiments must use ``level="l1"``.

    ``variant`` selects one of :data:`PROMPT_VARIANTS` (R171 only).  It rewords
    the instruction but keeps the definitions and the JSON contract, so Kappa
    differences are attributable to wording.
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}")
    opts = PROMPT_VARIANTS[variant]
    body: list[str] = [DEFINITIONS_L1 if level == "l1" else DEFINITIONS]
    if level == "l1" and opts.get("preface"):
        body.insert(0, opts["preface"])
    if level == "l1" and opts.get("rules"):
        body.append(opts["rules"])
    if extra_senses:
        body.append("本语料中出现的义项：" + "、".join(sorted(extra_senses)))
    if regime in {"few_shot", "cot"}:
        shots = list(shots)
        if shots:
            body.append(_example_block(shots, level=level))
    body.append(
        _render_explicit(item, level=level)
        if item.task == "explicit_sense"
        else _render_implicit(item, level=level)
    )

    if regime == "cot":
        body.append(
            '先简要说明判断依据（不超过 40 字），再给出标签。输出格式：\n'
            '{"理由": "...", "sense": "<义项>"}'
        )
    elif level == "l1":
        body.append(
            '输出格式（严格 JSON）：'
            '{"sense": "<TEMPORAL|CONTINGENCY|COMPARISON|EXPANSION>"}'
        )
    else:
        body.append('输出格式（严格 JSON）：{"sense": "<义项>"}')

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n\n".join(body)},
    ]


_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.S)
_SENSE_IN_TEXT = re.compile(
    r"\b(TEMPORAL|CONTINGENCY|COMPARISON|EXPANSION)(?::[A-Za-z][A-Za-z \-]*(?::[A-Za-z][A-Za-z \-]*)?)?"
)


def parse_sense(raw: str, allowed: Sequence[str] | None = None) -> str:
    """Extract a sense label from a model response.

    Order: JSON ``sense`` field -> any JSON object with a sense-like value ->
    regex over the raw text.  Returns ``""`` when nothing is found.
    """
    if not raw:
        return ""
    for blob in _JSON_OBJ.findall(raw):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            for key in ("sense", "label", "关系", "义项", "答案"):
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    return _match_allowed(val.strip(), allowed)
    m = _SENSE_IN_TEXT.search(raw.upper())
    if m:
        return _match_allowed(m.group(0), allowed)
    return ""


def _match_allowed(value: str, allowed: Sequence[str] | None) -> str:
    """Snap a predicted sense onto the allowed inventory when possible."""
    value = value.strip().strip('"').strip("'")
    if not allowed:
        return value
    exact = {a.upper(): a for a in allowed}
    if value.upper() in exact:
        return exact[value.upper()]
    # Prefix match: model said CONTINGENCY:Cause, inventory has CONTINGENCY:Cause:Reason.
    cands = [a for a in allowed if a.upper().startswith(value.upper() + ":")]
    if len(cands) == 1:
        return cands[0]
    return value


def sense_inventory(items: Iterable[Item], level: str = "l3") -> list[str]:
    attr = {"l1": "sense_l1", "l2": "sense_l2", "l3": "sense_l3"}[level]
    return sorted({getattr(it, attr) for it in items if getattr(it, attr)})


def level_zh(l1: str) -> str:
    return L1_ZH.get(l1, l1)
