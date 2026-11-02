from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from feature_extractor import build_feature_frame
from generators.chart_generator import generate_risk_chart
from generators.poster_generator import generate_security_poster
from generators.report_generator import generate_detection_report
from generators.table_generator import export_detection_table
from model_service import predict_url, predict_urls
from prompts import (
    FINAL_RESPONSE_PROMPT,
    GENERAL_SECURITY_QA_PROMPT,
    INTENT_ROUTING_PROMPT,
    MISJUDGMENT_ANALYSIS_PROMPT,
    REPORT_GENERATION_PROMPT,
    RISK_EXPLANATION_PROMPT,
)
from workflow_state import AgentState

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency fallback
    ChatOpenAI = None
    ChatPromptTemplate = None
    StrOutputParser = None


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

URL_REGEX = re.compile(r"https?://[^\s，。；;]+", re.IGNORECASE)
BARE_DOMAIN_REGEX = re.compile(
    r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s，。；;]*)?", re.IGNORECASE
)


def _normalize_result_list(detection_results: Any) -> list[dict[str, Any]]:
    if not detection_results:
        return []
    if isinstance(detection_results, dict):
        return [detection_results]
    return list(detection_results)


def _extract_urls(text: str) -> list[str]:
    if not text:
        return []

    found: list[str] = []
    seen: set[str] = set()

    url_spans: list[tuple[int, int]] = []
    for match in URL_REGEX.finditer(text):
        cleaned = match.group(0).rstrip("),.。；;")
        if cleaned not in seen:
            seen.add(cleaned)
            found.append(cleaned)
        url_spans.append(match.span())

    for match in BARE_DOMAIN_REGEX.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in url_spans):
            continue
        candidate = match.group(0).rstrip("),.。；;")
        if candidate.startswith(("http://", "https://")):
            candidate = candidate
        else:
            candidate = f"https://{candidate}"
        if candidate not in seen:
            seen.add(candidate)
            found.append(candidate)

    return found


def _contains_keywords(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_greeting(text: str) -> bool:
    greeting_keywords = [
        "你好",
        "您好",
        "hello",
        "hi",
        "在吗",
        "嗨",
        "hey",
        "早上好",
        "下午好",
        "晚上好",
    ]
    return _contains_keywords(text, greeting_keywords) and len(text.strip()) <= 16


def _is_capability_query(text: str) -> bool:
    capability_keywords = [
        "你会干什么",
        "你能做什么",
        "有什么功能",
        "支持哪些任务",
        "你能帮我什么",
        "你可以做什么",
        "能做什么",
    ]
    return _contains_keywords(text, capability_keywords)


def _get_llm():
    if ChatOpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    options: dict[str, Any] = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": api_key,
        "temperature": 0.2,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        options["base_url"] = base_url
    return ChatOpenAI(**options)


def _run_llm(prompt_text: str, fallback: str) -> str:
    llm = _get_llm()
    if llm is None or ChatPromptTemplate is None or StrOutputParser is None:
        return fallback

    try:
        prompt = ChatPromptTemplate.from_messages([("user", prompt_text)])
        chain = prompt | llm | StrOutputParser()
        return str(chain.invoke({}))
    except Exception:
        return fallback


def _classify_with_llm(user_input: str, urls: list[str]) -> str | None:
    llm = _get_llm()
    if llm is None or ChatPromptTemplate is None or StrOutputParser is None:
        return None

    try:
        prompt = ChatPromptTemplate.from_template(INTENT_ROUTING_PROMPT)
        chain = prompt | llm | StrOutputParser()
        raw = str(chain.invoke({"user_input": user_input, "urls": urls}))
        candidate = raw.strip().split()[0].strip("`，。:：")
        allowed = {
            "detect_single",
            "detect_batch",
            "explain_last",
            "misjudgment_analysis",
            "generate_report",
            "export_table",
            "generate_chart",
            "generate_poster",
            "general_security_qa",
            "greeting",
            "capability_query",
            "no_url_guidance",
        }
        if candidate in allowed:
            return candidate
    except Exception:
        return None
    return None


def _build_reason_list(result: dict[str, Any]) -> list[str]:
    features = result.get("features", {}) or {}
    reasons: list[str] = []

    if not features.get("is_https", 0):
        reasons.append("URL 未使用 HTTPS，敏感信息传输的安全性较弱。")
    if features.get("is_domain_ip", 0):
        reasons.append("域名直接使用 IP 地址，钓鱼和临时伪装页面中较常见。")
    if int(features.get("subdomains", 0)) >= 2:
        reasons.append("子域名层级较多，可能存在域名包装或伪装意图。")
    if int(features.get("special_chars", 0)) >= 8:
        reasons.append("特殊字符较多，URL 结构看起来不够简洁。")
    if float(features.get("digit_to_length_ratio", 0)) >= 0.12:
        reasons.append("数字占比偏高，常见于随机化或混淆型链接。")
    if int(features.get("query_count", 0)) >= 2 or int(features.get("equals", 0)) >= 2:
        reasons.append("查询参数较多，可能用于携带跳转参数或追踪标记。")
    if int(features.get("path_depth", 0)) >= 4:
        reasons.append("路径层级较深，链接结构相对复杂。")
    if float(result.get("phishing_probability", 0)) >= 0.8:
        reasons.append("模型给出的钓鱼概率很高，说明整体特征组合与恶意 URL 更接近。")
    elif float(result.get("phishing_probability", 0)) >= 0.5:
        reasons.append("模型认为该链接具有中高风险特征，需要进一步核验。")

    if not reasons:
        reasons.append("当前 URL 在模型特征空间中表现出一定异常，建议继续核实来源。")

    return reasons[:5]


def _build_result_card(result: dict[str, Any]) -> str:
    return (
        f"URL: {result.get('url', '')}\n"
        f"预测类别: {result.get('label', '')}\n"
        f"钓鱼概率: {float(result.get('phishing_probability', 0)):.4f}\n"
        f"正常概率: {float(result.get('legitimate_probability', 0)):.4f}\n"
        f"风险等级: {result.get('risk_level', '')}\n"
        f"特征: {json.dumps(result.get('features', {}), ensure_ascii=False)}"
    )


def _build_batch_summary(results: list[dict[str, Any]]) -> str:
    total = len(results)
    phishing = sum(1 for item in results if item.get("prediction") == 1)
    legitimate = sum(1 for item in results if item.get("prediction") == 0)
    risk_counter = Counter(item.get("risk_level", "") for item in results)
    return (
        f"总样本数: {total}\n"
        f"钓鱼样本数: {phishing}\n"
        f"正常样本数: {legitimate}\n"
        f"低风险: {risk_counter.get('低风险', 0)}\n"
        f"中风险: {risk_counter.get('中风险', 0)}\n"
        f"高风险: {risk_counter.get('高风险', 0)}"
    )


def rule_based_intent(
    user_input: str, urls: list[str], last_detection_results: Any = None
) -> str:
    text = (user_input or "").strip().lower()
    has_urls = len(urls) > 0
    many_urls = (
        len(urls) >= 2 or "批量" in text or "下面这些" in text or "这些 url" in text
    )

    if _is_greeting(text):
        return "greeting"
    if _is_capability_query(text):
        return "capability_query"

    llm_intent = _classify_with_llm(user_input, urls)
    if llm_intent:
        return llm_intent

    if _contains_keywords(text, ["生成报告", "写成检测报告", "整理结果", "报告"]):
        return "generate_report"
    if _contains_keywords(
        text, ["导出表格", "保存为 csv", "保存为 excel", "生成检测结果表", "导出结果表"]
    ):
        return "export_table"
    if _contains_keywords(
        text,
        [
            "生成图表",
            "画图",
            "可视化",
            "风险图",
            "概率图",
            "生成风险表图",
            "画风险表图",
            "生成风险概率图",
            "生成风险等级分布图",
            "检测结果图",
            "风险概率图",
        ],
    ):
        return "generate_chart"
    if _contains_keywords(
        text, ["生成安全提示图", "生成防钓鱼宣传图", "生成图片提醒", "安全提示图片"]
    ):
        return "generate_poster"
    if _contains_keywords(
        text, ["什么是钓鱼网站", "如何识别钓鱼链接", "网络安全", "钓鱼网站", "恶意 url"]
    ):
        return "general_security_qa"
    if _contains_keywords(
        text, ["误判", "准确吗", "会不会错", "模型为什么会错", "可能误判"]
    ):
        return "misjudgment_analysis"
    if _contains_keywords(
        text, ["为什么", "依据是什么", "哪里可疑", "高风险原因", "为什么高风险"]
    ):
        if last_detection_results:
            return "explain_last"
    if many_urls:
        return "detect_batch"
    if has_urls:
        return "detect_single"
    if _contains_keywords(text, ["检测", "看看", "安全吗", "安全不安全", "帮我看看"]):
        return "no_url_guidance"
    if last_detection_results and _contains_keywords(text, ["为什么", "依据", "原因"]):
        return "explain_last"
    return "general_security_qa"


def classify_intent(
    user_input: str, urls: list[str], last_detection_results: Any = None
) -> str:
    return rule_based_intent(user_input, urls, last_detection_results)


def _make_llm_text(
    prompt_template: str, variables: dict[str, Any], fallback: str
) -> str:
    llm = _get_llm()
    if llm is None or ChatPromptTemplate is None or StrOutputParser is None:
        return fallback

    try:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        return str(chain.invoke(variables))
    except Exception:
        return fallback


def intent_router_node(state: AgentState) -> AgentState:
    user_input = state.get("user_input", "")
    last_detection_results = state.get("last_detection_results")
    urls = _extract_urls(user_input)
    intent = rule_based_intent(user_input, urls, last_detection_results)
    return {"intent": intent, "urls": urls}


def extract_urls_node(state: AgentState) -> AgentState:
    urls = state.get("urls") or _extract_urls(state.get("user_input", ""))
    return {"urls": urls}


def no_url_guidance_node(state: AgentState) -> AgentState:
    state["final_answer"] = (
        "你希望我帮你检测网址，但当前输入里没有识别到可检测的 URL。\n\n"
        "请直接发给我完整链接，例如：\n"
        "- https://www.baidu.com\n"
        "- http://login-paypal-security-check.xyz\n\n"
        "如果是批量检测，也可以把多个 URL 直接换行发给我。"
    )
    return state


def greeting_node(state: AgentState) -> AgentState:
    state["final_answer"] = (
        "你好，我可以帮你做这些事：\n"
        "1. 检测单个 URL 是否有钓鱼风险；\n"
        "2. 批量检测多个 URL；\n"
        "3. 解释风险原因和误判可能；\n"
        "4. 生成检测报告、表格和风险图表；\n"
        "5. 生成防钓鱼安全提示图；\n"
        "6. 回答网络安全相关问题。\n\n"
        "你可以直接发一个链接给我，或者说“生成风险图表”“导出表格”“生成安全提示图”。"
    )
    return state


def capability_node(state: AgentState) -> AgentState:
    state["final_answer"] = (
        "我可以把钓鱼网站检测做成一个完整工作流，而不只是单次预测。\n\n"
        "支持的功能包括：\n"
        "- 单个 URL 检测；\n"
        "- 批量 URL 检测；\n"
        "- 风险原因解释；\n"
        "- 误判分析；\n"
        "- 生成检测报告；\n"
        "- 导出 CSV / Excel 表格；\n"
        "- 生成风险图表；\n"
        "- 生成防钓鱼安全提示图；\n"
        "- 回答网络安全相关问题。\n\n"
        "如果你愿意，我可以直接帮你检测一个网址，或者演示批量检测。"
    )
    return state


def single_detection_node(state: AgentState) -> AgentState:
    urls = state.get("urls") or []
    if not urls:
        state["error"] = "没有可检测的 URL。"
        return state

    result = predict_url(urls[0])
    result["risk_reasons"] = _build_reason_list(result)
    state["detection_results"] = [result]
    state["last_detection_results"] = [result]
    return state


def batch_detection_node(state: AgentState) -> AgentState:
    urls = state.get("urls") or []
    if not urls:
        state["error"] = "没有可检测的 URL。"
        return state

    results = predict_urls(urls)
    for result in results:
        result["risk_reasons"] = _build_reason_list(result)
    state["detection_results"] = results
    state["last_detection_results"] = results
    return state


def risk_explanation_node(state: AgentState) -> AgentState:
    results = _normalize_result_list(
        state.get("detection_results") or state.get("last_detection_results")
    )
    if not results:
        state["explanations"] = "没有可用于解释的检测结果，请先执行 URL 检测。"
        return state

    if len(results) == 1:
        result = results[0]
        phishing_probability = float(result.get("phishing_probability", 0))
        if phishing_probability < 0.3:
            fallback = "\n".join(
                [
                    f"该 URL 被判定为 {result.get('label', '')}，钓鱼概率为 {phishing_probability:.4f}，整体属于低风险。",
                    "主要判断依据如下：",
                    *[f"- {reason}" for reason in _build_reason_list(result)[:3]],
                    f"安全建议：{result.get('suggestion', '')}",
                ]
            )
        else:
            fallback = "\n".join(
                [
                    f"该 URL 被判定为 {result.get('label', '')}，钓鱼概率为 {phishing_probability:.4f}。",
                    "主要风险原因如下：",
                    *[f"- {reason}" for reason in _build_reason_list(result)],
                    f"安全建议：{result.get('suggestion', '')}",
                ]
            )
        text = _make_llm_text(
            RISK_EXPLANATION_PROMPT,
            {
                "detection_result": _build_result_card(result),
                "features": json.dumps(result.get("features", {}), ensure_ascii=False),
            },
            fallback,
        )
        state["explanations"] = text
        return state

    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. {result.get('url', '')} -> {result.get('label', '')}，钓鱼概率 {float(result.get('phishing_probability', 0)):.4f}，风险等级 {result.get('risk_level', '')}"
        )
    fallback = "\n".join(
        [
            "批量检测结果的风险解释如下：",
            *lines,
            "建议优先关注高风险链接，并结合域名、证书与访问场景复核。",
        ]
    )
    state["explanations"] = fallback
    return state


def misjudgment_analysis_node(state: AgentState) -> AgentState:
    results = _normalize_result_list(
        state.get("last_detection_results") or state.get("detection_results")
    )
    if not results:
        state["final_answer"] = "当前没有可用于误判分析的检测结果，请先检测一个 URL。"
        return state

    result = results[0]
    fallback = "\n".join(
        [
            f"该结果存在误判可能，原因包括：",
            "- 模型只基于 URL 特征，不看网页内容和证书细节。",
            "- 某些正常网站也可能包含较长路径、较多参数或多级子域名。",
            "- 某些钓鱼网站可能使用正常外观的短链接或较干净的 URL 结构。",
            f"本次结果：{result.get('label', '')}，钓鱼概率 {float(result.get('phishing_probability', 0)):.4f}。",
        ]
    )
    text = _make_llm_text(
        MISJUDGMENT_ANALYSIS_PROMPT,
        {
            "detection_result": _build_result_card(result),
            "features": json.dumps(result.get("features", {}), ensure_ascii=False),
        },
        fallback,
    )
    state["final_answer"] = text
    return state


def report_generation_node(state: AgentState) -> AgentState:
    results = _normalize_result_list(
        state.get("last_detection_results") or state.get("detection_results")
    )
    if not results:
        state["error"] = "没有可用于生成报告的检测结果。"
        return state

    output_format = (
        "txt"
        if _contains_keywords(
            state.get("user_input", "").lower(), ["txt", "文本", "纯文本"]
        )
        else "md"
    )
    state["report_path"] = generate_detection_report(
        results, output_format=output_format
    )
    return state


def table_export_node(state: AgentState) -> AgentState:
    results = _normalize_result_list(
        state.get("last_detection_results") or state.get("detection_results")
    )
    if not results:
        state["error"] = "没有可用于导出表格的检测结果。"
        return state

    output_format = (
        "excel"
        if _contains_keywords(
            state.get("user_input", "").lower(), ["excel", "xlsx", "xls"]
        )
        else "csv"
    )
    state["table_path"] = export_detection_table(results, output_format=output_format)
    return state


def chart_generation_node(state: AgentState) -> AgentState:
    results = _normalize_result_list(
        state.get("last_detection_results") or state.get("detection_results")
    )
    if not results:
        state["error"] = "没有可用于生成图表的检测结果。"
        return state

    state["chart_path"] = generate_risk_chart(results)
    return state


def poster_generation_node(state: AgentState) -> AgentState:
    state["poster_path"] = generate_security_poster()
    return state


def general_security_qa_node(state: AgentState) -> AgentState:
    user_input = state.get("user_input", "")
    fallback_map = [
        (
            ["什么是钓鱼网站", "钓鱼网站是什么", "恶意 url"],
            "钓鱼网站通常会伪装成银行、支付、邮箱、社交平台等正规站点，诱导用户输入账号、密码或验证码。",
        ),
        (
            ["如何识别", "怎么识别", "识别钓鱼"],
            "可以重点看域名是否拼写异常、是否有多级子域名、是否频繁跳转，以及是否诱导你输入敏感信息。",
        ),
        (
            ["防范", "怎么防", "如何防范"],
            "最稳妥的做法是从官网入口访问，不随意点陌生链接，不在可疑页面输入密码或验证码。",
        ),
        (
            ["误判", "准确", "为什么会错"],
            "这类模型只看 URL 结构，不看网页内容，所以少量误判是正常的，最好结合上下文人工复核。",
        ),
    ]

    user_lower = user_input.lower()
    base = None
    for keywords, answer in fallback_map:
        if _contains_keywords(user_lower, keywords):
            base = answer
            break
    if base is None:
        base = "你可以把问题问得更具体一点，比如“什么是钓鱼网站”“如何识别可疑链接”“这个模型为什么会误判”。"

    fallback = base
    state["final_answer"] = _make_llm_text(
        GENERAL_SECURITY_QA_PROMPT + "\n用户问题：{user_input}",
        {"user_input": user_input},
        fallback,
    )
    return state


def response_generation_node(state: AgentState) -> AgentState:
    results = _normalize_result_list(
        state.get("detection_results") or state.get("last_detection_results")
    )
    explanation = state.get("explanations", "")

    if not results:
        state["final_answer"] = "没有可展示的检测结果，请先输入一个 URL 进行检测。"
        return state

    if len(results) == 1:
        result = results[0]
        reasons = _build_reason_list(result)
        reason_text = "\n".join(
            f"{index + 1}. {reason}" for index, reason in enumerate(reasons)
        )
        fallback = (
            "【URL 安全检测结果】\n\n"
            f"检测对象：{result.get('url', '')}\n"
            f"预测类别：{result.get('label', '')}\n"
            f"钓鱼风险概率：{float(result.get('phishing_probability', 0)):.4f}\n"
            f"正常概率：{float(result.get('legitimate_probability', 0)):.4f}\n"
            f"风险等级：{result.get('risk_level', '')}\n\n"
            f"主要风险原因：\n{reason_text}\n\n"
            f"安全建议：{result.get('suggestion', '')}\n\n"
            "你可以继续输入：\n"
            "- 为什么它是这个风险等级？\n"
            "- 这个结果可能误判吗？\n"
            "- 生成检测报告\n"
            "- 导出表格\n"
            "- 生成风险图表\n"
            "- 生成安全提示图"
        )
        state["final_answer"] = _make_llm_text(
            FINAL_RESPONSE_PROMPT
            + "\n检测结果：{detection_result}\n解释：{explanations}",
            {
                "detection_result": _build_result_card(result),
                "explanations": explanation,
            },
            fallback,
        )
        return state

    summary = _build_batch_summary(results)
    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. {result.get('url', '')} -> {result.get('label', '')}，钓鱼概率 {float(result.get('phishing_probability', 0)):.4f}，风险等级 {result.get('risk_level', '')}"
        )
    fallback = (
        "【批量 URL 安全检测结果】\n\n"
        f"{summary}\n\n"
        "结果明细：\n" + "\n".join(lines) + "\n\n你可以继续输入：\n"
        "- 为什么它是高风险？\n"
        "- 这个结果可能误判吗？\n"
        "- 生成检测报告\n"
        "- 导出表格\n"
        "- 生成风险图表\n"
        "- 生成安全提示图"
    )
    state["final_answer"] = fallback
    return state


def file_response_node(state: AgentState) -> AgentState:
    if state.get("report_path"):
        state["final_answer"] = (
            "【文件生成完成】\n\n"
            "文件类型：检测报告\n"
            f"保存路径：{state['report_path']}\n\n"
            "说明：该文件基于最近一次检测结果生成。"
        )
        return state
    if state.get("table_path"):
        state["final_answer"] = (
            "【文件生成完成】\n\n"
            "文件类型：检测结果表格\n"
            f"保存路径：{state['table_path']}\n\n"
            "说明：该文件基于最近一次检测结果生成。"
        )
        return state
    state["final_answer"] = "没有可返回的文件路径。"
    return state


def image_response_node(state: AgentState) -> AgentState:
    if state.get("chart_path"):
        state["final_answer"] = (
            "【文件生成完成】\n\n"
            "文件类型：风险图表\n"
            f"保存路径：{state['chart_path']}\n\n"
            "说明：该图片基于最近一次检测结果生成。"
        )
        return state
    if state.get("poster_path"):
        state["final_answer"] = (
            "【文件生成完成】\n\n"
            "文件类型：防钓鱼安全提示图片\n"
            f"保存路径：{state['poster_path']}\n\n"
            "说明：该图片用于课程展示和安全宣传。"
        )
        return state
    state["final_answer"] = "没有可返回的图片路径。"
    return state
