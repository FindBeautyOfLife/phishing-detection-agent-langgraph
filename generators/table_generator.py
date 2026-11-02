from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_results(detection_results: Any) -> list[dict[str, Any]]:
    if detection_results is None:
        return []
    if isinstance(detection_results, dict):
        return [detection_results]
    return list(detection_results)


def _frame_from_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "url": result.get("url", ""),
                "label": result.get("label", ""),
                "phishing_probability": float(result.get("phishing_probability", 0)),
                "legitimate_probability": float(
                    result.get("legitimate_probability", 0)
                ),
                "risk_level": result.get("risk_level", ""),
                "suggestion": result.get("suggestion", ""),
            }
        )
    return pd.DataFrame(rows)


def export_detection_table(detection_results: Any, output_format: str = "csv") -> str:
    _ensure_output_dir()
    results = _normalize_results(detection_results)
    if not results:
        raise ValueError("没有可用于导出表格的检测结果。")

    frame = _frame_from_results(results)
    suffix = "xlsx" if output_format.lower() in {"excel", "xlsx", "xls"} else "csv"
    path = OUTPUT_DIR / f"url_detection_results_{_timestamp()}.{suffix}"

    if suffix == "csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        frame.to_excel(path, index=False)

    return str(path)
