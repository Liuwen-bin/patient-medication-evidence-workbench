from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel


EVIDENCE_SCOPE = re.compile(
    r"(?:(?:核查|查找|对照|复核|展示|查阅|检查|检索|提取|列出|整理|"
    r"review|find|show|check|compare|retrieve|extract|list|summarize)"
    r".{0,80}(?:标签|证据|原文|章节|说明书|处方|用药|药品|药物|医嘱|产品|"
    r"成分|途径|剂型|剂量|给药|储存|存储|过敏|特殊人群|妊娠|"
    r"section|evidence|label|information|prescription|product|medication|medicine|"
    r"drug|order|ingredient|route|dosage|dose|storage|pregnancy)|"
    r"\b(?:medication|medicine|drug|product)\b.{0,40}"
    r"\bfor\s+(?:the\s+)?(?:evidence|label|review)\b|"
    r"(?:标签|证据|原文|章节|说明书|处方|用药|药品|药物|医嘱|产品|"
    r"成分|途径|剂型|剂量|给药|储存|存储|过敏|特殊人群|妊娠)"
    r".{0,60}(?:核查|复核|审查))",
    re.IGNORECASE,
)
EVIDENCE_OBJECT = re.compile(
    r"标签|证据|原文|章节|说明书|警告|"
    r"\b(?:section|evidence|label|warning|prescribing\s+information)\b",
    re.IGNORECASE,
)
SAFE_EVIDENCE_NOUN_PHRASE = re.compile(
    r"^(?=.{1,180}$)(?:[a-z0-9'_-]+\s+){0,8}"
    r"(?:section\s+evidence|evidence|prescribing\s+information)"
    r"(?:\s+(?:from|in|for)\s+(?:the\s+)?"
    r"(?:label|pharmacist|this\s+product)){0,2}[.!?]?$",
    re.IGNORECASE,
)

_ZH_ACTION = (
    r"(?:停药|停用(?:这个|该|当前|上述)?药?|停止(?:服用|使用|吃|用药)(?:这个|该|当前|上述)?药?|"
    r"(?:换成|换用|改用|改吃|更换为)(?:(?:另一个|其他|别的|新)?药|[\u4e00-\u9fffA-Za-z0-9-]{1,20})|换药|"
    r"更换(?:这个|该|当前)?药|"
    r"推荐(?:一种|一个|其他|别的|新)?药|"
    r"(?:加倍|增加|加大|提高|减少|减小|降低|上调|下调|调整|改变|修改|减半)(?:这个|我的|该药的)?剂量|"
    r"(?:这个|我的|该药的)?剂量(?:加倍|增加|加大|提高|减少|减小|降低|上调|下调|调整|改变|修改|减半)|"
    r"(?:吃|服用|口服|使用|注射|贴敷)\s*(?:多少|几)\s*"
    r"(?:毫克|克|微克|毫升|片|粒|袋|支|次)|"
    r"(?:(?:每天|每日|每次|早晨|上午|中午|下午|晚上|睡前)\s*)+"
    r"(?:服用|口服|使用|注射)\s*(?:\d+(?:\.\d+)?|[一二两三四五六七八九十半]+)\s*"
    r"(?:毫克|克|微克|毫升|片|粒|袋|支|次)|"
     r"(?:服用|口服|使用|注射)\s*(?:\d+(?:\.\d+)?|[一二两三四五六七八九十半]+)\s*"
     r"(?:毫克|克|微克|毫升|片|粒|袋|支|次)(?:\s*(?:每天|每日|每次|早晨|上午|中午|下午|晚上|睡前))?|"
     r"(?:(?:每天|每日|每次|早晨|上午|中午|下午|晚上|睡前)\s*)?"
     r"(?:服用|口服|使用|注射|贴敷)\s*(?:这个|该|当前|上述)?"
     r"(?:药物?|[\u4e00-\u9fffA-Za-z0-9-]{2,30})|"
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
_EN_ADMIN_ACTION = (
    r"(?:take|use|administer|inject|apply)\b"
    r"(?![^.!?;]{0,80}\b(?:section|evidence|label|information|warning|warnings)\b)"
)
_EN_ACTION = (
    rf"(?:{_EN_STOP_ACTION}|{_EN_SWITCH_ACTION}|"
    rf"{_EN_ADMIN_ACTION}|"
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
    rf"\bwhat\s+dos(?:e|age)\s+should\s+"
    rf"(?:(?:this|the)\s+)?patient\s+(?:receive|take|use)\b|"
    rf"(?:^|[.!?;])\s*(?:please\s+)?(?:choose|select)\s+(?:a\s+)?"
    rf"(?:treatment|therapy|drug|medication|medicine)\s+for\s+"
    rf"(?:(?:this|the)\s+)?patient\b|"
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
TREATMENT_DECISION = re.compile(
    r"(?:决定|确定|选择|制定|推荐).{0,60}(?:治疗|疗法|诊疗方案)|"
    r"(?:患者|我).{0,30}(?:应该|应当|需要|该).{0,30}(?:如何治疗|接受什么治疗|用什么疗法)|"
    r"\b(?:decide|determine|choose|select|recommend)\b.{0,80}"
    r"\b(?:treat(?:ment|ed)|therapy)\b|"
    r"\b(?:what|which)\s+(?:treatment|therapy)\b.{0,60}"
    r"\b(?:patient|me|i)\b",
    re.IGNORECASE,
)
PATIENT_DECISION_OUTPUT = re.compile(
    r"\b(?:decide|determine|pick|select|choose|work\s+out)\b"
    r"(?=[^.!?;]{0,100}\b(?:whether|if|best|safe|treat(?:ment|ed)|therapy|"
    r"care\s+plan|regimen|dose|dosage)\b)|"
    r"\b(?:tell|advise|recommend)\s+(?:me|(?:(?:this|the)\s+)?patient)\b"
    r"(?=[^.!?;]{0,100}\b(?:whether|if|should|must|needs?|safe|best|continue|"
    r"take|use|receive|treat(?:ment|ed)|therapy|care\s+plan|regimen|dose|dosage)\b)|"
    r"\bhave\s+(?:(?:this|the)\s+)?patient\s+"
    r"(?:continue|take|use|receive|start|stop|switch|change)\b|"
    r"\b(?:say|state|answer|assess|evaluate|judge)\b"
    r"(?=[^.!?;]{0,120}\b(?:whether|if|best|safe|appropriate|suitable|ought|"
    r"should|may|can|continue|stay\s+on|keep\s+taking|dose|dosage|regimen|"
    r"care\s+plan|treat(?:ment|ed)|therapy)\b)|"
    r"\b(?:is|are)\s+(?:this|the|my|your|his|her)?\s*"
    r"(?:medication|medicine|drug|dose|dosage|regimen|treatment|therapy)\b"
    r"[^.!?;]{0,40}\b(?:safe|appropriate|suitable|best)\b|"
    r"(?:判断|评估|评价|决定|确定|回答).{0,80}(?:"
    r"(?:这个|该|当前)?(?:药|药物|剂量|疗法|治疗).{0,30}(?:是否)?"
    r"(?:适合|安全|恰当|最佳)|"
    r"(?:这个|该|当前)?患者.{0,30}(?:是否|应该|应当|可以|能否).{0,30}"
    r"(?:继续|服药|用药|停药|换药|调整剂量))",
    re.IGNORECASE,
)
PATIENT_SUITABILITY_DECISION = re.compile(
    r"(?:(?=[^.!?;。！？；]{0,180}\b(?:patient|me|i)\b)"
    r"(?=[^.!?;。！？；]{0,180}\b(?:medication|medicine|drug|dose|dosage|"
    r"regimen|treatment|therapy|use)\b)"
    r"(?=[^.!?;。！？；]{0,180}\b(?:safe|appropriate|acceptable|suitable|"
    r"makes\s+sense)\b)[^.!?;。！？；]{1,180}|"
    r"(?=[^.!?;。！？；]{0,120}(?:患者|我))"
    r"(?=[^.!?;。！？；]{0,120}(?:药|药物|剂量|用药|治疗|疗法))"
    r"(?=[^.!?;。！？；]{0,120}(?:安全|适合|合适|恰当|可接受|妥(?:不妥)?))"
    r"[^.!?;。！？；]{1,120})",
    re.IGNORECASE,
)
CONTINUED_THERAPY_DECISION = re.compile(
    r"(?:(?=[^.!?;。！？；]{0,180}\b(?:patient|i|me)\b)"
    r"(?=[^.!?;。！？；]{0,180}\b(?:continue|continuing|ongoing\s+use|"
    r"keep\s+taking|stay\s+on)\b)"
    r"(?=[^.!?;。！？；]{0,180}\b(?:may|can|could|should|whether|if|"
    r"acceptable|safe|appropriate|suitable|makes\s+sense)\b)"
    r"[^.!?;。！？；]{1,180}|"
    r"(?=[^.!?;。！？；]{0,120}患者)"
    r"(?=[^.!?;。！？；]{0,120}继续(?:用药|服药|服用|使用|"
    r"吃(?:这个|该|当前|上述)?药|用))"
    r"(?=[^.!?;。！？；]{0,120}(?:能不能|能否|是否|可以|应该|应当|"
    r"该|能|应|安全|适合))"
    r"[^.!?;。！？；]{1,120})",
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
        "OUT_OF_SCOPE_REVIEW_QUESTION",
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
        EVIDENCE_SCOPE.search(normalized)
        or SAFE_EVIDENCE_NOUN_PHRASE.fullmatch(normalized)
    )
    explicit_action = bool(
        DIRECTED_CLINICAL_ACTION.search(normalized)
        or TREATMENT_DECISION.search(normalized)
        or PATIENT_DECISION_OUTPUT.search(normalized)
        or PATIENT_SUITABILITY_DECISION.search(normalized)
        or CONTINUED_THERAPY_DECISION.search(normalized)
    )
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
    if not evidence_scope:
        return QuestionSafetyDecision(
            allowed=False,
            code="OUT_OF_SCOPE_REVIEW_QUESTION",
            explanation="请求不属于药师用药证据核查范围，未调用患者或药品数据工具。",
        )
    return QuestionSafetyDecision(
        allowed=True,
        code="ALLOWED_EVIDENCE_REVIEW",
        explanation="请求属于药师证据核查范围。",
    )
