from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "reports"


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


def _format_reason_list(result: dict[str, Any]) -> list[str]:
    reasons = result.get("risk_reasons") or result.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    return [str(reason) for reason in reasons if str(reason).strip()]


def _batch_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    phishing = sum(1 for item in results if item.get("prediction") == 1)
    legitimate = sum(1 for item in results if item.get("prediction") == 0)
    return {
        "total": total,
        "phishing": phishing,
        "legitimate": legitimate,
        "high": sum(1 for item in results if item.get("risk_level") == "高风险"),
        "medium": sum(1 for item in results if item.get("risk_level") == "中风险"),
        "low": sum(1 for item in results if item.get("risk_level") == "低风险"),
    }


def _render_single_md(result: dict[str, Any]) -> str:
    reasons = _format_reason_list(result)
    reasons_md = (
        "\n".join(f"{index + 1}. {reason}" for index, reason in enumerate(reasons))
        or "1. 模型特征组合显示存在异常模式"
    )

    return f"""# 钓鱼网站检测报告

## 检测概况
- 检测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 检测对象：{result.get('url', '')}
- 预测类别：{result.get('label', '')}
- 钓鱼概率：{result.get('phishing_probability', 0):.4f}
- 正常概率：{result.get('legitimate_probability', 0):.4f}
- 风险等级：{result.get('risk_level', '')}

## 主要风险原因
{reasons_md}

## 安全建议
{result.get('suggestion', '建议谨慎访问该网址。')}

## 模型局限性说明
1. 模型基于 URL 结构化特征判断，无法直接解析页面内容、JS 行为或实时跳转链。
2. 结果依赖训练数据分布，对于新型混淆手法和极端样本可能存在误判。
3. 若 URL 被短链接、重定向或中间跳转包装，最终风险还需结合人工复核。
"""


def _render_batch_md(results: list[dict[str, Any]]) -> str:
    summary = _batch_summary(results)
    rows = []
    for index, result in enumerate(results, start=1):
        rows.append(
            f"| {index} | {result.get('url', '')} | {result.get('label', '')} | {result.get('phishing_probability', 0):.4f} | {result.get('legitimate_probability', 0):.4f} | {result.get('risk_level', '')} | {result.get('suggestion', '')} |"
        )

    return f"""# 批量 URL 检测报告

## 检测概况
- 检测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 总样本数：{summary['total']}
- 钓鱼样本数：{summary['phishing']}
- 正常样本数：{summary['legitimate']}
- 高风险：{summary['high']}
- 中风险：{summary['medium']}
- 低风险：{summary['low']}

## 结果明细
| 序号 | URL | 预测类别 | 钓鱼概率 | 正常概率 | 风险等级 | 建议 |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## 模型局限性说明
1. 批量检测适合初筛，不应替代对高价值目标的人工安全审查。
2. 模型仅使用 URL 结构特征，无法识别页面可视内容和后台真实跳转链。
3. 若样本分布与训练数据差异较大，概率输出需要结合场景解释。
"""


def _render_single_txt(result: dict[str, Any]) -> str:
    reasons = _format_reason_list(result)
    reason_lines = (
        "\n".join(f"{index + 1}. {reason}" for index, reason in enumerate(reasons))
        or "1. 模型特征组合显示存在异常模式"
    )

    return f"""【钓鱼网站检测报告】

检测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
检测对象：{result.get('url', '')}
预测类别：{result.get('label', '')}
钓鱼概率：{result.get('phishing_probability', 0):.4f}
正常概率：{result.get('legitimate_probability', 0):.4f}
风险等级：{result.get('risk_level', '')}

主要风险原因：
{reason_lines}

安全建议：
{result.get('suggestion', '建议谨慎访问该网址。')}

模型局限性说明：
1. 模型仅基于 URL 特征进行判断，不能替代人工安全分析。
2. 无法直接识别网页内容、证书状态和完整跳转行为。
3. 对于短链接、重定向链接和包装链接，需结合上下文复核。
"""


def _render_batch_txt(results: list[dict[str, Any]]) -> str:
    summary = _batch_summary(results)
    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. {result.get('url', '')} | {result.get('label', '')} | 钓鱼概率 {result.get('phishing_probability', 0):.4f} | 风险等级 {result.get('risk_level', '')}"
        )

    return f"""【批量 URL 检测报告】

检测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
总样本数：{summary['total']}
钓鱼样本数：{summary['phishing']}
正常样本数：{summary['legitimate']}
高风险：{summary['high']}
中风险：{summary['medium']}
低风险：{summary['low']}

结果明细：
{chr(10).join(lines)}

模型局限性说明：
1. 批量结果适合初筛，不应直接替代人工复核。
2. 模型只看 URL 特征，不看完整网页内容。
3. 对高风险结果建议结合域名、页面和访问场景进一步确认。
"""


def generate_detection_report(detection_results: Any, output_format: str = "md") -> str:
    _ensure_output_dir()
    results = _normalize_results(detection_results)
    if not results:
        raise ValueError("没有可用于生成报告的检测结果。")

    suffix = "md" if output_format.lower() not in {"txt", "text"} else "txt"
    path = OUTPUT_DIR / f"url_detection_report_{_timestamp()}.{suffix}"

    if len(results) == 1:
        content = (
            _render_single_md(results[0])
            if suffix == "md"
            else _render_single_txt(results[0])
        )
    else:
        content = (
            _render_batch_md(results) if suffix == "md" else _render_batch_txt(results)
        )

    path.write_text(content, encoding="utf-8")
    return str(path)
