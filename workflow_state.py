from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    intent: str
    urls: list[str]
    detection_results: list[dict[str, Any]] | dict[str, Any]
    last_detection_results: list[dict[str, Any]] | dict[str, Any] | None
    explanations: str | list[str] | list[dict[str, Any]]
    report_path: str
    table_path: str
    chart_path: str
    poster_path: str
    final_answer: str
    error: str
