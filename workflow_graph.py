from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from workflow_nodes import (
    batch_detection_node,
    chart_generation_node,
    capability_node,
    extract_urls_node,
    file_response_node,
    general_security_qa_node,
    image_response_node,
    intent_router_node,
    misjudgment_analysis_node,
    no_url_guidance_node,
    poster_generation_node,
    greeting_node,
    report_generation_node,
    response_generation_node,
    risk_explanation_node,
    single_detection_node,
    table_export_node,
)
from workflow_state import AgentState


def _route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "")
    if intent == "greeting":
        return "greeting_node"
    if intent == "capability_query":
        return "capability_node"
    if intent in {"detect_single", "detect_batch"}:
        return "extract_urls_node"
    if intent == "explain_last":
        return "risk_explanation_node"
    if intent == "misjudgment_analysis":
        return "misjudgment_analysis_node"
    if intent == "generate_report":
        return "report_generation_node"
    if intent == "export_table":
        return "table_export_node"
    if intent == "generate_chart":
        return "chart_generation_node"
    if intent == "generate_poster":
        return "poster_generation_node"
    if intent == "general_security_qa":
        return "general_security_qa_node"
    return "no_url_guidance_node"


def _route_after_extract(state: AgentState) -> str:
    intent = state.get("intent", "")
    urls = state.get("urls") or []
    if not urls:
        return "no_url_guidance_node"
    if intent == "detect_batch" or len(urls) >= 2:
        return "batch_detection_node"
    return "single_detection_node"


def build_workflow_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent_router_node", intent_router_node)
    graph.add_node("extract_urls_node", extract_urls_node)
    graph.add_node("no_url_guidance_node", no_url_guidance_node)
    graph.add_node("greeting_node", greeting_node)
    graph.add_node("capability_node", capability_node)
    graph.add_node("single_detection_node", single_detection_node)
    graph.add_node("batch_detection_node", batch_detection_node)
    graph.add_node("risk_explanation_node", risk_explanation_node)
    graph.add_node("misjudgment_analysis_node", misjudgment_analysis_node)
    graph.add_node("report_generation_node", report_generation_node)
    graph.add_node("table_export_node", table_export_node)
    graph.add_node("chart_generation_node", chart_generation_node)
    graph.add_node("poster_generation_node", poster_generation_node)
    graph.add_node("general_security_qa_node", general_security_qa_node)
    graph.add_node("response_generation_node", response_generation_node)
    graph.add_node("file_response_node", file_response_node)
    graph.add_node("image_response_node", image_response_node)

    graph.set_entry_point("intent_router_node")

    graph.add_conditional_edges(
        "intent_router_node",
        _route_after_intent,
        {
            "extract_urls_node": "extract_urls_node",
            "greeting_node": "greeting_node",
            "capability_node": "capability_node",
            "risk_explanation_node": "risk_explanation_node",
            "misjudgment_analysis_node": "misjudgment_analysis_node",
            "report_generation_node": "report_generation_node",
            "table_export_node": "table_export_node",
            "chart_generation_node": "chart_generation_node",
            "poster_generation_node": "poster_generation_node",
            "general_security_qa_node": "general_security_qa_node",
            "no_url_guidance_node": "no_url_guidance_node",
        },
    )

    graph.add_conditional_edges(
        "extract_urls_node",
        _route_after_extract,
        {
            "single_detection_node": "single_detection_node",
            "batch_detection_node": "batch_detection_node",
            "no_url_guidance_node": "no_url_guidance_node",
        },
    )

    graph.add_edge("single_detection_node", "risk_explanation_node")
    graph.add_edge("batch_detection_node", "risk_explanation_node")
    graph.add_edge("risk_explanation_node", "response_generation_node")
    graph.add_edge("greeting_node", END)
    graph.add_edge("capability_node", END)
    graph.add_edge("report_generation_node", "file_response_node")
    graph.add_edge("table_export_node", "file_response_node")
    graph.add_edge("chart_generation_node", "image_response_node")
    graph.add_edge("poster_generation_node", "image_response_node")

    graph.add_edge("no_url_guidance_node", END)
    graph.add_edge("misjudgment_analysis_node", END)
    graph.add_edge("general_security_qa_node", END)
    graph.add_edge("response_generation_node", END)
    graph.add_edge("file_response_node", END)
    graph.add_edge("image_response_node", END)

    return graph.compile()


WORKFLOW_APP = build_workflow_graph()


def run_workflow(user_input: str, last_detection_results: Any = None) -> dict[str, Any]:
    initial_state: AgentState = {
        "user_input": user_input,
        "last_detection_results": last_detection_results,
    }
    return WORKFLOW_APP.invoke(initial_state)
