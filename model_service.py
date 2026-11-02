from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from feature_extractor import FEATURE_NAMES, build_feature_frame

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_names.pkl"


@lru_cache(maxsize=1)
def load_model() -> tuple[Any, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到模型文件：{MODEL_PATH}")
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"未找到特征文件：{FEATURE_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    if not isinstance(feature_names, list):
        feature_names = list(feature_names)
    return model, feature_names


def _get_model_classes(model: Any) -> list[Any]:
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps") and model.named_steps:
        last_step = list(model.named_steps.values())[-1]
        if hasattr(last_step, "classes_"):
            return list(last_step.classes_)
    raise AttributeError("模型不包含 classes_ 属性，无法提取类别概率。")


def get_class_probability(model: Any, X: pd.DataFrame, target_class: int) -> float:
    probabilities = model.predict_proba(X)[0]
    class_index = _get_model_classes(model).index(target_class)
    return float(probabilities[class_index])


def get_risk_level(phishing_probability: float) -> str:
    if phishing_probability < 0.3:
        return "低风险"
    if phishing_probability < 0.7:
        return "中风险"
    return "高风险"


def _label_from_prediction(prediction: int) -> str:
    return "钓鱼/恶意 URL" if int(prediction) == 1 else "正常 URL"


def _build_suggestion(risk_level: str) -> str:
    if risk_level == "高风险":
        return "建议立即停止访问，不要输入账号、密码或验证码。"
    if risk_level == "中风险":
        return "建议核对域名、证书和页面内容，确认来源后再继续。"
    return "整体风险较低，但仍建议确认网址拼写和访问来源。"


def predict_url(url: str) -> dict[str, Any]:
    model, feature_names = load_model()
    X = build_feature_frame([url], feature_names)

    prediction = int(model.predict(X)[0])
    phishing_probability = get_class_probability(model, X, 1)
    legitimate_probability = get_class_probability(model, X, 0)
    risk_level = get_risk_level(phishing_probability)

    return {
        "url": url,
        "prediction": prediction,
        "label": _label_from_prediction(prediction),
        "phishing_probability": float(phishing_probability),
        "legitimate_probability": float(legitimate_probability),
        "risk_level": risk_level,
        "suggestion": _build_suggestion(risk_level),
        "features": X.iloc[0].to_dict(),
        "feature_names": feature_names or FEATURE_NAMES,
    }


def predict_urls(urls: list[str]) -> list[dict[str, Any]]:
    return [predict_url(url) for url in urls]
