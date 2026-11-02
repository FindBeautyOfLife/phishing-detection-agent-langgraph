# -*- coding: utf-8 -*-

"""
train_model.py

基于 URL 特征的钓鱼网站检测模型训练脚本。

功能：
1. 读取 data/ 文件夹下的 CSV 数据集；
2. 自动识别 URL 列和 label 列；
3. 保持原始标签不反转；
4. 从 URL 中提取结构化特征；
5. 训练多个机器学习模型；
6. 使用钓鱼类 F1-score 选择最佳模型；
7. 保存模型和特征名称。

标签约定：
1 = 钓鱼 / 恶意 URL
0 = 正常 URL
"""

import os
import re
import warnings
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")


# =========================
# 1. 基本配置
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = BASE_DIR / "phishing_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_names.pkl"
RESULT_PATH = BASE_DIR / "model_comparison_results.csv"
VISUALIZATION_DIR = BASE_DIR / "outputs" / "training_visualizations"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# 每个 CSV 文件最多抽样 5000 条，加快训练
MAX_ROWS_PER_FILE = 5000


# =========================
# 2. URL 特征配置
# =========================

FEATURE_NAMES = [
    "dots",
    "at",
    "equals",
    "slashes",
    "hyphens",
    "colons",
    "question_marks",
    "digits",
    "underscore",
    "tilde",
    "percent",
    "lowercase",
    "uppercase",
    "upper_to_lower_ratio",
    "is_https",
    "url_length",
    "domain_length",
    "path_length",
    "path_depth",
    "query_length",
    "query_count",
    "fragment_length",
    "is_domain_ip",
    "is_tld_iana_reg",
    "is_mtld",
    "subdomains",
    "special_chars",
    "digit_to_length_ratio",
    "char_to_length_ratio",
    "specialchar_to_length_ratio",
]


# =========================
# 3. URL 特征提取
# =========================


def is_ip_address(domain: str) -> int:
    """
    判断域名是否是 IPv4 地址。
    """
    pattern = r"^\d{1,3}(\.\d{1,3}){3}$"
    return int(bool(re.match(pattern, domain)))


def extract_url_features(url: str) -> dict:
    """
    从 URL 中提取结构化特征。
    """

    if pd.isna(url):
        url = ""

    url = str(url).strip()

    if url.startswith(("http://", "https://")):
        parsed_url = url
    else:
        parsed_url = "https://" + url

    parsed = urlparse(parsed_url)

    domain = parsed.netloc
    path = parsed.path
    query = parsed.query
    fragment = parsed.fragment

    url_length = len(url)

    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    lowercase = sum(c.islower() for c in url)
    uppercase = sum(c.isupper() for c in url)
    special_chars = sum(not c.isalnum() for c in url)

    tld = domain.split(".")[-1].lower() if "." in domain else ""

    common_tlds = {
        "com",
        "cn",
        "org",
        "net",
        "edu",
        "gov",
        "io",
        "co",
        "jp",
        "uk",
        "de",
        "fr",
        "ru",
        "us",
        "info",
        "biz",
    }

    multi_tlds = {"com.cn", "edu.cn", "gov.cn", "co.uk", "ac.uk"}

    last_two_parts = (
        ".".join(domain.split(".")[-2:]).lower() if domain.count(".") >= 1 else ""
    )

    features = {
        "dots": url.count("."),
        "at": url.count("@"),
        "equals": url.count("="),
        "slashes": url.count("/"),
        "hyphens": url.count("-"),
        "colons": url.count(":"),
        "question_marks": url.count("?"),
        "digits": digits,
        "underscore": url.count("_"),
        "tilde": url.count("~"),
        "percent": url.count("%"),
        "lowercase": lowercase,
        "uppercase": uppercase,
        "upper_to_lower_ratio": uppercase / lowercase if lowercase else 0,
        "is_https": int(url.startswith("https://")),
        "url_length": url_length,
        "domain_length": len(domain),
        "path_length": len(path),
        "path_depth": path.count("/"),
        "query_length": len(query),
        "query_count": query.count("="),
        "fragment_length": len(fragment),
        "is_domain_ip": is_ip_address(domain),
        "is_tld_iana_reg": int(tld in common_tlds),
        "is_mtld": int(last_two_parts in multi_tlds),
        "subdomains": max(domain.count(".") - 1, 0),
        "special_chars": special_chars,
        "digit_to_length_ratio": digits / url_length if url_length else 0,
        "char_to_length_ratio": letters / url_length if url_length else 0,
        "specialchar_to_length_ratio": special_chars / url_length if url_length else 0,
    }

    return features


# =========================
# 4. 数据读取与标签处理
# =========================


def find_column(df: pd.DataFrame, candidates: list[str], col_type: str) -> str:
    """
    自动查找 URL 列或 label 列。
    """
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(f"未找到 {col_type} 列，请检查数据集列名。")


def normalize_label(label):
    """
    保持原始标签语义，不进行标签反转。

    标签约定：
    1 = 钓鱼 / 恶意 URL
    0 = 正常 URL
    """
    if pd.isna(label):
        return np.nan

    try:
        value = int(float(label))
        return value if value in [0, 1] else np.nan
    except Exception:
        return np.nan


def load_dataset(data_dir: str) -> pd.DataFrame:
    """
    读取 data 文件夹下所有 CSV 文件，并合并为一个数据集。
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据文件夹不存在：{data_dir}")

    csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(f"{data_dir} 文件夹下没有 CSV 文件。")

    print(f"发现 {len(csv_files)} 个 CSV 文件：")
    for file in csv_files:
        print(f" - {file}")

    data_list = []

    for file in csv_files:
        file_path = os.path.join(data_dir, file)

        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin1")

        if len(df) > MAX_ROWS_PER_FILE:
            df = df.sample(n=MAX_ROWS_PER_FILE, random_state=RANDOM_STATE)

        url_col = find_column(
            df, ["url", "URL", "Url", "link", "Link", "website", "Website"], "URL"
        )

        label_col = find_column(
            df, ["label", "Label", "class", "Class", "target", "Target"], "label"
        )

        temp = df[[url_col, label_col]].copy()
        temp.columns = ["url", "label"]

        data_list.append(temp)

    data = pd.concat(data_list, ignore_index=True)

    data["label"] = data["label"].apply(normalize_label)
    data = data.dropna(subset=["url", "label"])
    data["label"] = data["label"].astype(int)

    data = data.drop_duplicates(subset=["url"]).reset_index(drop=True)

    print("\n数据读取完成")
    print(f"总样本数：{len(data)}")
    print("标签分布：")
    print(data["label"].value_counts())

    print("\n标签含义：")
    print("1 = 钓鱼 / 恶意 URL")
    print("0 = 正常 URL")

    return data


# =========================
# 5. 构建特征矩阵
# =========================


def build_feature_dataset(data: pd.DataFrame):
    """
    将 URL 转换为模型训练特征。
    """

    print("\n开始提取 URL 特征...")

    feature_dicts = data["url"].apply(extract_url_features)
    X = pd.DataFrame(feature_dicts.tolist())
    X = X.reindex(columns=FEATURE_NAMES, fill_value=0)

    y = data["label"]

    print("特征提取完成")
    print(f"特征数量：{len(FEATURE_NAMES)}")

    return X, y


# =========================
# 6. 模型定义
# =========================


def get_models():
    """
    定义候选模型。
    """

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=63,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=10,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    return models


# =========================
# 7. 模型训练与评估
# =========================


def evaluate_model(model, X_test, y_test) -> dict:
    """
    评估模型性能。
    重点关注钓鱼类，即 label = 1。
    """

    y_pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Phishing Precision": precision_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "Phishing Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Phishing F1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
    }


def train_and_select_best_model(X, y):
    """
    训练多个模型，并选择钓鱼类 F1-score 最高的模型。
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("\n训练集 / 测试集划分完成")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")

    models = get_models()
    trained_models = {}
    results = []

    for name, model in models.items():
        print(f"\n正在训练：{name}")

        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics = evaluate_model(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)

        print(f"Accuracy           : {metrics['Accuracy']:.4f}")
        print(f"Phishing Precision : {metrics['Phishing Precision']:.4f}")
        print(f"Phishing Recall    : {metrics['Phishing Recall']:.4f}")
        print(f"Phishing F1        : {metrics['Phishing F1']:.4f}")

        print("\n分类报告：")
        print(
            classification_report(
                y_test,
                model.predict(X_test),
                labels=[0, 1],
                target_names=["Legitimate", "Phishing"],
                digits=4,
            )
        )

    result_df = pd.DataFrame(results)
    result_df = result_df[
        ["Model", "Accuracy", "Phishing Precision", "Phishing Recall", "Phishing F1"]
    ].sort_values(by="Phishing F1", ascending=False)

    best_model_name = result_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print("\n模型对比结果：")
    print(result_df)

    print(f"\n最佳模型：{best_model_name}")

    y_test_best = y_test
    y_pred_best = best_model.predict(X_test)

    y_proba_best = None
    if hasattr(best_model, "predict_proba"):
        y_proba_best = best_model.predict_proba(X_test)[:, 1]

    return (
        best_model,
        best_model_name,
        result_df,
        y_test_best,
        y_pred_best,
        y_proba_best,
    )


def save_visualizations(
    result_df: pd.DataFrame,
    best_model_name: str,
    y_test,
    y_pred,
    y_proba,
    output_dir: str = VISUALIZATION_DIR,
):
    """
    保存训练结果可视化图像。
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1) 模型对比图（Phishing F1）
    sorted_df = result_df.sort_values(by="Phishing F1", ascending=False)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_df["Model"], sorted_df["Phishing F1"], color="#2E8B57")
    plt.title("Model Comparison - Phishing F1")
    plt.xlabel("Model")
    plt.ylabel("Phishing F1")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    model_compare_path = os.path.join(output_dir, "model_f1_comparison.png")
    plt.savefig(model_compare_path, dpi=150)
    plt.close()

    # 2) 最佳模型混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Legitimate", "Phishing"])
    plt.yticks(tick_marks, ["Legitimate", "Phishing"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "best_model_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 3) ROC 曲线
    roc_path = None
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(
            fpr, tpr, color="#D2691E", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {best_model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()

        roc_path = os.path.join(output_dir, "best_model_roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()

    print("\n可视化图像保存完成：")
    print(f"- {model_compare_path}")
    print(f"- {cm_path}")
    if roc_path:
        print(f"- {roc_path}")


# =========================
# 8. 保存结果
# =========================


def save_artifacts(best_model, result_df):
    """
    保存模型、特征名和模型评估结果。
    """

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(FEATURE_NAMES, FEATURE_PATH)
    result_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    print("\n文件保存完成：")
    print(f"- {MODEL_PATH}")
    print(f"- {FEATURE_PATH}")
    print(f"- {RESULT_PATH}")


# =========================
# 9. 概率提取函数
# =========================


def get_class_probability(model, X, target_class: int) -> float:
    """
    从 predict_proba 中提取指定类别的概率。
    这样比直接写 proba[1] 更稳妥。
    """

    proba = model.predict_proba(X)[0]
    class_index = list(model.classes_).index(target_class)
    return float(proba[class_index])


# =========================
# 10. 主函数
# =========================


def main():
    print("=" * 60)
    print("基于机器学习的钓鱼网站检测模型训练")
    print("=" * 60)

    data = load_dataset(DATA_DIR)

    X, y = build_feature_dataset(data)

    (
        best_model,
        best_model_name,
        result_df,
        y_test,
        y_pred,
        y_proba,
    ) = train_and_select_best_model(X, y)

    save_visualizations(result_df, best_model_name, y_test, y_pred, y_proba)

    save_artifacts(best_model, result_df)

    print("\n训练流程结束。")


if __name__ == "__main__":
    main()
