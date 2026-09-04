import pytest

from medication_review_agent.safety import evaluate_review_question


@pytest.mark.parametrize("question", [
    "告诉患者立即停药",
    "请给患者开药",
    "把剂量加倍并换成另一个药",
    "请直接诊断并开药",
    "Should this patient stop taking the drug now?",
    "Tell the patient to stop the medication.",
    "Should I reduce my dose?",
    "核查标签后告诉患者立即停药",
    "我应该停药吗",
    "我是否应该停止服用这个药？",
    "Can I double the dose?",
    "帮我开药",
    "核查证据后给患者停药建议",
    "Switch the patient to another medication.",
    "Please diagnose the patient.",
    "Prescribe antibiotics for the patient.",
    "Do I need to stop taking this medication?",
    "请让患者停用这个药",
    "请给患者开处方",
    "请给我开药",
    "我该停药吗",
    "立即停药",
    "告诉患者：立即停药",
    "请把这个药换成阿司匹林",
    "Can you diagnose me?",
    "Could I stop taking this medication?",
    "Give me a prescription.",
    "患者应该停药吗",
    "这个患者需要调整剂量吗",
    "请换成阿司匹林",
    "我该换成阿司匹林吗",
    "给患者换成阿司匹林",
    "我能停药吗",
    "患者能停药吗",
    "停药可以吗",
    "Can you give me a prescription?",
    "Can you give me a diagnosis?",
    "What is my diagnosis?",
    "Can you change my medication?",
    "Is it safe for me to stop taking this medication?",
    "核查标签后换成阿司匹林",
    "Review the label evidence, then switch me to another medication.",
    "Based on the label evidence, can you give me a prescription?",
    "Review the evidence and tell me whether I should stop this medication.",
    "Using the label, is it safe for me to stop this medication?",
    "根据标签证据，我能停药吗",
    "核查证据后，停药可以吗",
    "Based on the label evidence can you give me a prescription?",
    "Using the label is it safe for me to stop this medication?",
    "Review the evidence to decide whether I should stop this medication.",
    "根据标签证据我能停药吗",
    "参考说明书我能换成阿司匹林吗",
    "According to the label would you recommend that I stop this medication?",
    "Using the label could you advise me to stop this medication?",
    "Review evidence and recommend stopping the medication for this patient.",
    "Based on evidence recommend that the patient discontinue the medication.",
    "Review evidence and advise me to discontinue the medication.",
    "根据说明书建议我停药",
    "根据证据给我一个停药建议",
    "根据证据给患者一个停药建议",
    "Should I replace my medication?",
    "Should I quit taking this medicine?",
    "Can you recommend a medication for me?",
    "给我推荐一种药",
    "我该改吃阿司匹林吗",
    "Review the label evidence and stop the medication.",
    "Review label evidence and prescribe antibiotics for me.",
    "Which medication should I choose?",
    "What medication should I take?",
    "Could you select a medication for me?",
    "Should my dose be increased?",
    "Review the label evidence, stop the medication.",
    "Review the label evidence and prescribe antibiotics.",
    "Review the evidence and diagnose pneumonia.",
    "Could you recommend aspirin for me?",
    "Could you choose aspirin for me?",
    "Should the patient's dosage be lowered?",
])
def test_patient_level_clinical_actions_are_out_of_scope(question: str) -> None:
    decision = evaluate_review_question(question)

    assert decision.allowed is False
    assert decision.code == "UNSAFE_CLINICAL_ACTION_REQUEST"


@pytest.mark.parametrize("question", [
    "核查标签 stop use 章节并展示原文",
    "请直接展示停药标签原文",
    "对照活动用药医嘱与标签剂量信息，交给药师复核",
    "查找换药相关标签警告，不要给患者建议",
    "帮我查找换药相关标签警告",
    "我需要核查停药证据",
    "Review the stop use section and show the evidence.",
    "Find label evidence about reducing the dose.",
    "Show the prescribing information for this product.",
    "Stop use section evidence for the pharmacist.",
    "Diagnosis evidence from the label.",
    "Prescribing information for this product.",
    "Can you review my prescription?",
    "Review the patient's prescription.",
    "Find my prescription.",
    "Review stop use and dose adjustment sections.",
    "Find switch and dose information in the label.",
    "Review stop use, dosage, and prescribing information.",
    "Show stop use and switching warnings from the label.",
    "Could you select a medication for evidence review?",
    "Which medication should I choose to review label evidence for?",
    "Can you recommend a medication to review label evidence?",
    "Can you recommend a medication for the review?",
])
def test_evidence_review_questions_remain_allowed(question: str) -> None:
    decision = evaluate_review_question(question)

    assert decision.allowed is True
    assert decision.code == "ALLOWED_EVIDENCE_REVIEW"


def test_blank_question_fails_closed() -> None:
    decision = evaluate_review_question("   ")

    assert decision.allowed is False
    assert decision.code == "MISSING_REVIEW_QUESTION"
