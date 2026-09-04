from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel


EVIDENCE_SCOPE = re.compile(
    r"(?:核查|查找|对照|复核|展示|review|find|show)"
    r".{0,80}(?:标签|证据|原文|章节|说明书|处方|section|evidence|label|information|prescription)",
    re.IGNORECASE,
)
EVIDENCE_OBJECT = re.compile(
    r"标签|证据|原文|章节|说明书|警告|"
    r"\b(?:section|evidence|label|warning|prescribing\s+information|information)\b",
    re.IGNORECASE,
)

_ZH_ACTION = (
    r"(?:停药|停用(?:这个|该|当前|上述)?药?|停止(?:服用|使用|吃|用药)(?:这个|该|当前|上述)?药?|"
    r"(?:换成|换用|改用|改吃|更换为)(?:(?:另一个|其他|别的|新)?药|[\u4e00-\u9fffA-Za-z0-9-]{1,20})|换药|"
    r"更换(?:这个|该|当前)?药|"
    r"推荐(?:一种|一个|其他|别的|新)?药|"
    r"(?:加倍|增加|加大|提高|减少|减小|降低|上调|下调|调整|改变|修改|减半)(?:这个|我的|该药的)?剂量|"
    r"(?:这个|我的|该药的)?剂量(?:加倍|增加|加大|提高|减少|减小|降低|上调|下调|调整|改变|修改|减半)|"
    r"诊断|确诊|开药|开(?:具)?(?:药物|用药)?处方)"
)
_EN_STOP_ACTION = (
    r"(?:stop|discontinue|quit)\b(?:"
    r"\s+taking\b(?:\s+(?:the|this|my|your|his|her))?(?:\s+(?:drug|medication|medicine))?|"
    r"\s+(?:the|this|my|your|his|her)?\s*(?:drug|medication|medicine)\b|"
    r"(?=\s*(?:(?:now|immediately)\s*)?(?:[.!?;]|$)))"
)
_EN_SWITCH_ACTION = (
    r"(?:switch|change|replace)\b(?:"
    r"\s+(?:me|(?:(?:this|the)\s+)?patient)\b|"
    r"\s+(?:my|the|this|your|his|her|patient'?s)?\s*"
    r"(?:drug|medication|medicine|medications|medicines)\b|"
    r"\s+to\b|(?=\s*(?:[.!?;]|$)))"
)
_EN_EVIDENCE_PURPOSE = (
    r"(?:to\s+(?:review|find|show)\b|"
    r"for\s+(?:the\s+)?(?:evidence|label|review)\b)"
)
_EN_ACTION = (
    rf"(?:{_EN_STOP_ACTION}|{_EN_SWITCH_ACTION}|"
    r"recommend\s+(?:(?:a|another|some)\s+)?(?:drug|medication|medicine)\b"
    rf"(?!\s+{_EN_EVIDENCE_PURPOSE})"
    r"(?:\s+for\s+(?:me|(?:(?:this|the)\s+)?patient))?|"
    r"(?:increase|reduce|decrease|lower|adjust|change|double|halve)"
    r"\b(?:\s+(?:the|this|my|your|his|her|patient'?s))?\s+dos(?:e|age)\b|"
    r"diagnose\b|prescribe\b)"
)
_EN_GERUND_ACTION = (
    r"(?:(?:stopping|discontinuing)(?:\s+taking)?(?:\s+(?:the|this))?"
    r"(?:\s+(?:drug|medication|medicine))?|"
    r"(?:switching|changing)(?:\s+(?:the|this|my|patient'?s))?"
    r"\s+(?:drug|medication|medicine)|"
    r"(?:increasing|reducing|decreasing|lowering|adjusting|doubling|halving)"
    r"(?:\s+(?:the|this|my|patient'?s))?\s+dos(?:e|age))"
)

DIRECTED_CLINICAL_ACTION = re.compile(
    rf"(?:"
    rf"(?:告诉|告知|建议|要求|让|叫|指导|嘱咐)(?:这位|该|当前|这个)?患者"
    rf"\s*[:：，,]?\s*(?:立即|马上|现在|应该|应当|需要|必须|可以)?{_ZH_ACTION}|"
    rf"(?:给|为)(?:(?:这位|该|当前|这个)?患者|我)(?:制定|提供|做出)?{_ZH_ACTION}(?:建议|方案|指示)?|"
    rf"(?:建议|推荐)(?:我|(?:这位|该|当前|这个)?患者)(?:立即|马上|现在)?{_ZH_ACTION}|"
    rf"(?:给|为)(?:我|(?:这位|该|当前|这个)?患者)(?:一个|一份)?"
    rf"{_ZH_ACTION}(?:的)?(?:建议|方案)|"
    rf"(?:这位|该|当前|这个)?患者(?:是否)?"
    rf"(?:应该|应当|需要|必须|能否|能|可以|要不要|该不该|该)"
    rf"(?:立即|马上|现在)?{_ZH_ACTION}|"
    rf"我(?:是否)?(?:应该|应当|需要|必须|能否|能|可以|要不要|该不该|该)"
    rf"(?:立即|马上|现在)?{_ZH_ACTION}|"
    rf"{_ZH_ACTION}(?:是否)?(?:可以|能|应该|应当)吗|"
    rf"(?:后|然后|随后|再)\s*(?:请)?(?:立即|马上|现在)?{_ZH_ACTION}|"
    rf"(?:^|[，。！？；])\s*(?:请(?:直接)?|直接|帮我)(?:立即|马上|现在)?{_ZH_ACTION}|"
    rf"(?:^|[，。！？；])\s*(?:立即|马上|现在){_ZH_ACTION}|"
    rf"(?:^|[，。！？；])\s*(?:请(?:直接)?|直接|帮我)?把"
    rf"(?:这位|该|当前|这个)?患者(?:立即|马上|现在)?{_ZH_ACTION}|"
    rf"(?:^|[，。！？；])\s*(?:请(?:直接)?|直接|帮我)?把"
    rf"(?:这个|我的|该药的)?剂量(?:立即|马上|现在)?"
    rf"(?:加倍|增加|加大|提高|减少|减小|降低|上调|下调|调整|改变|修改|减半)|"
    rf"(?:^|[，。！？；])\s*(?:请(?:直接)?|直接|帮我)?把"
    rf"(?:这个|该|当前|上述)?药(?:换成|改成|更换为|改用).{{1,20}}(?:[，。！？；]|$)|"
    rf"(?:tell|advise|ask|instruct)\s+(?:(?:this|the)\s+)?patient\s+to\s+{_EN_ACTION}|"
    rf"(?:should|must|can|could|may)\s+(?:i|(?:(?:this|the)\s+)?patient)\s+{_EN_ACTION}|"
    rf"(?:can|could|would|will)\s+you\s+(?:please\s+)?{_EN_ACTION}|"
    rf"(?:do|did)\s+i\s+(?:need|have)\s+to\s+{_EN_ACTION}|"
    rf"\bi\s+(?:should|must|can|could|may|need\s+to|have\s+to)\s+{_EN_ACTION}|"
    rf"\bis\s+it\s+(?:safe|okay|ok)\s+for\s+"
    rf"(?:me|(?:(?:this|the)\s+)?patient)\s+to\s+{_EN_ACTION}|"
    rf"(?:(?:this|the)\s+)?patient\s+(?:should|must|can|needs?\s+to)\s+{_EN_ACTION}|"
    rf"(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:give|write|get)\s+"
    rf"(?:me|(?:(?:this|the)\s+)?patient)\s+(?:a\s+)?(?:prescription|diagnosis)\b|"
    rf"(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    rf"(?:recommend|choose|select)\b"
    rf"(?![^.!?;]{{0,80}}\b{_EN_EVIDENCE_PURPOSE})"
    rf"[^.!?;]{{1,80}}\b(?:me|(?:(?:this|the)\s+)?patient)\b|"
    rf"(?:^|[.!?;])\s*(?:please\s+)?(?:give|write|get)\s+"
    rf"(?:me|(?:(?:this|the)\s+)?patient)\s+(?:a\s+)?(?:prescription|diagnosis)\b|"
    rf"\bwhat\s+is\s+(?:my|(?:(?:this|the)\s+)?patient'?s)\s+diagnosis\b|"
    rf"\b(?:which|what)\s+(?:drug|medication|medicine)\s+should\s+"
    rf"(?:i|(?:(?:this|the)\s+)?patient)\s+(?:choose|select|take)\b"
    rf"(?!\s+{_EN_EVIDENCE_PURPOSE})|"
    rf"\b(?:should|must|can|could|may)\s+"
    rf"(?:my|the\s+patient'?s|the|this|your|his|her|patient'?s)"
    rf"\s+dos(?:e|age)\s+be\s+"
    rf"(?:increased|reduced|decreased|lowered|adjusted|changed|doubled|halved)\b|"
    rf"\b(?:recommend|advise)\s+(?:"
    rf"(?:that\s+)?(?:i|(?:(?:this|the)\s+)?patient)\s+(?:should\s+)?{_EN_ACTION}|"
    rf"(?:me|(?:(?:this|the)\s+)?patient)\s+to\s+{_EN_ACTION}|"
    rf"{_EN_GERUND_ACTION}(?:\s+for\s+(?:me|(?:(?:this|the)\s+)?patient))?"
    rf")|"
    rf"\b(?:and\s+)?then\s+(?:please\s+)?{_EN_ACTION}|"
    rf"(?:\band\b|[,;])\s*(?:please\s+)?{_EN_ACTION}"
    rf"(?!\s+(?:section|evidence|information|warning|warnings)\b)|"
    rf"(?:^|[.!?;])\s*(?:please\s+)?{_EN_ACTION}"
    rf")",
    re.IGNORECASE,
)

CLINICAL_ACTION_TOPIC = re.compile(
    r"停药|停用|停止(?:服用|使用|吃|用药)|换药|换成|换用|改用|改吃|更换|"
    r"剂量.{0,8}(?:加倍|增加|加大|提高|减少|减小|降低|上调|下调|调整|改变|修改|减半)|"
    r"(?:加倍|增加|加大|提高|减少|减小|降低|上调|下调|调整|改变|修改|减半).{0,8}剂量|"
    r"诊断|确诊|开药|处方|"
    r"\b(?:stop|discontinue|quit|switch|change|replace|increase|reduce|decrease|lower|adjust|double|halve|"
    r"diagnose|diagnosis|prescribe|prescription)\b",
    re.IGNORECASE,
)
PATIENT_REQUEST_CONTEXT = re.compile(
    r"我|患者|请|立即|马上|现在|可以吗|能吗|应该吗|应当吗|需要吗|"
    r"\b(?:i|me|my|you|your|patient)\b",
    re.IGNORECASE,
)
CLAUSE_BOUNDARY = re.compile(
    r"\s*(?:[。.！!?？；;]|然后|随后|之后|后|再|并且|\b(?:and\s+then|then)\b)\s*",
    re.IGNORECASE,
)


class QuestionSafetyDecision(BaseModel):
    allowed: bool
    code: Literal[
        "ALLOWED_EVIDENCE_REVIEW",
        "UNSAFE_CLINICAL_ACTION_REQUEST",
        "MISSING_REVIEW_QUESTION",
    ]
    explanation: str


def evaluate_review_question(question: str) -> QuestionSafetyDecision:
    normalized = " ".join(question.casefold().split())
    if not normalized:
        return QuestionSafetyDecision(
            allowed=False,
            code="MISSING_REVIEW_QUESTION",
            explanation="缺少可审核的问题，未调用患者或药品数据工具。",
        )
    evidence_scope = bool(
        EVIDENCE_SCOPE.search(normalized) or EVIDENCE_OBJECT.search(normalized)
    )
    explicit_action = bool(DIRECTED_CLINICAL_ACTION.search(normalized))
    contextual_action = False
    for clause in CLAUSE_BOUNDARY.split(normalized):
        action = CLINICAL_ACTION_TOPIC.search(clause)
        if action is None:
            continue
        clause_is_evidence = bool(
            EVIDENCE_SCOPE.search(clause) or EVIDENCE_OBJECT.search(clause)
        )
        direct_context = bool(PATIENT_REQUEST_CONTEXT.search(clause))
        if not clause_is_evidence and (direct_context or action.start() <= 8):
            contextual_action = True
            break
    directed_action = explicit_action or contextual_action
    if directed_action:
        return QuestionSafetyDecision(
            allowed=False,
            code="UNSAFE_CLINICAL_ACTION_REQUEST",
            explanation="系统只能整理证据并交由药师审核，不能给出患者级诊疗动作。",
        )
    return QuestionSafetyDecision(
        allowed=True,
        code="ALLOWED_EVIDENCE_REVIEW",
        explanation=(
            "请求属于药师证据核查范围。"
            if evidence_scope
            else "未检测到明确的患者级诊疗动作。"
        ),
    )
