from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "charts"


def _configure_fonts() -> None:
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]

    for font_path in font_candidates:
        path = Path(font_path)
        if not path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
            font_name = font_manager.FontProperties(fname=str(path)).get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue

    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_fonts()


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


def _abbreviate_url(url: str, max_len: int = 24) -> str:
    if len(url) <= max_len:
        return url
    return f"{url[:16]}...{url[-10:]}"


def _risk_color(risk_level: str) -> str:
    colors = {
        "低风险": "#2f9e6d",
        "中风险": "#d69028",
        "高风险": "#b4232a",
    }
    return colors.get(risk_level, "#6b7280")


def _style_axes(ax) -> None:
    ax.set_facecolor("#fffdf8")
    ax.grid(axis="x", color="#e7ded2", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d8c8b8")
    ax.spines["bottom"].set_color("#d8c8b8")
    ax.tick_params(colors="#40342f", labelsize=10)


def _add_probability_label(ax, x: float, y: float, label: str) -> None:
    label_x = min(max(x + 0.025, 0.08), 0.93)
    ax.text(
        label_x,
        y,
        label,
        va="center",
        ha="left",
        fontsize=10,
        color="#2e2522",
        fontweight="bold",
    )


def _save_chart(fig, path: Path) -> None:
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), pad_inches=0.25)
    plt.close(fig)


def generate_risk_chart(detection_results: Any) -> str:
    _ensure_output_dir()
    results = _normalize_results(detection_results)
    if not results:
        raise ValueError("没有可用于生成图表的检测结果。")

    path = OUTPUT_DIR / f"risk_chart_{_timestamp()}.png"

    if len(results) == 1:
        result = results[0]
        phishing_prob = float(result.get("phishing_probability", 0))
        legitimate_prob = float(result.get("legitimate_probability", 0))
        risk_level = str(result.get("risk_level", "未知风险"))
        url = _abbreviate_url(str(result.get("url", "")), max_len=42)

        fig, ax = plt.subplots(figsize=(9, 5.4))
        fig.patch.set_facecolor("#f7f3ec")
        ax.set_facecolor("#fffdf8")
        fig.text(
            0.08,
            0.93,
            "单个 URL 风险概率图",
            ha="left",
            fontsize=19,
            fontweight="bold",
            color="#241b18",
        )
        fig.text(0.08, 0.86, url, ha="left", fontsize=10, color="#6b5750")
        fig.text(
            0.08,
            0.78,
            f"风险等级：{risk_level}",
            ha="left",
            fontsize=13,
            fontweight="bold",
            color=_risk_color(risk_level),
        )

        labels = ["钓鱼概率", "正常概率"]
        values = [phishing_prob, legitimate_prob]
        colors = ["#b4232a", "#2f9e6d"]
        ax.barh(labels, values, color=colors, height=0.42)
        ax.set_xlim(0, 1)
        ax.set_xlabel("概率", color="#40342f")
        _style_axes(ax)
        for index, value in enumerate(values):
            _add_probability_label(ax, value, index, f"{value:.1%}")
        ax.invert_yaxis()
        fig.subplots_adjust(left=0.18, right=0.94, top=0.64, bottom=0.18)
        _save_chart(fig, path)
        return str(path)

    urls = [str(item.get("url", "")) for item in results]
    phishing_probs = [float(item.get("phishing_probability", 0)) for item in results]
    risk_levels = [str(item.get("risk_level", "低风险")) for item in results]
    risk_counts = {"低风险": 0, "中风险": 0, "高风险": 0}
    for item in results:
        risk_counts[str(item.get("risk_level", "低风险"))] = (
            risk_counts.get(str(item.get("risk_level", "低风险")), 0) + 1
        )

    sorted_rows = sorted(
        zip(urls, phishing_probs, risk_levels), key=lambda row: row[1], reverse=True
    )
    urls, phishing_probs, risk_levels = map(list, zip(*sorted_rows))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, max(5.8, 0.72 * len(urls) + 2.9)),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )
    fig.patch.set_facecolor("#f7f3ec")
    fig.text(
        0.06,
        0.94,
        "批量 URL 风险检测图表",
        ha="left",
        fontsize=21,
        fontweight="bold",
        color="#241b18",
    )
    fig.text(
        0.06,
        0.885,
        "按钓鱼概率从高到低排序，颜色表示风险等级",
        ha="left",
        fontsize=11,
        color="#6b5750",
    )

    y_positions = list(range(len(urls)))
    bar_colors = [_risk_color(level) for level in risk_levels]
    axes[0].barh(y_positions, phishing_probs, color=bar_colors, height=0.48)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([_abbreviate_url(url, 36) for url in urls])
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("钓鱼概率", color="#40342f")
    axes[0].set_title("URL 钓鱼概率对比", loc="left", fontsize=14, fontweight="bold")
    axes[0].axvline(0.3, color="#d69028", linestyle="--", linewidth=1)
    axes[0].axvline(0.7, color="#b4232a", linestyle="--", linewidth=1)
    _style_axes(axes[0])
    axes[0].invert_yaxis()
    for y, value in zip(y_positions, phishing_probs):
        _add_probability_label(axes[0], value, y, f"{value:.1%}")

    labels = list(risk_counts.keys())
    values = [risk_counts[label] for label in labels]
    axes[1].bar(labels, values, color=[_risk_color(label) for label in labels], width=0.55)
    axes[1].set_title("风险等级分布", loc="left", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("数量", color="#40342f")
    axes[1].set_ylim(0, max(values + [1]) * 1.25)
    axes[1].grid(axis="y", color="#e7ded2", linewidth=1)
    axes[1].grid(axis="x", visible=False)
    _style_axes(axes[1])
    for index, value in enumerate(values):
        axes[1].text(
            index,
            value + max(values + [1]) * 0.04,
            str(value),
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="#2e2522",
        )

    legend_handles = [
        Patch(facecolor=_risk_color("低风险"), label="低风险 < 30%"),
        Patch(facecolor=_risk_color("中风险"), label="中风险 30%-70%"),
        Patch(facecolor=_risk_color("高风险"), label="高风险 >= 70%"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.035),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.23, right=0.95, top=0.80, bottom=0.12, wspace=0.3)
    _save_chart(fig, path)
    return str(path)
