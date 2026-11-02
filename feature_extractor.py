from __future__ import annotations

import re
from urllib.parse import urlparse

import pandas as pd

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


def is_ip_address(domain: str) -> int:
    pattern = r"^\d{1,3}(\.\d{1,3}){3}$"
    return int(bool(re.match(pattern, domain or "")))


def _normalize_url(url: str) -> tuple[str, str, str, str, str]:
    if pd.isna(url):
        url = ""

    url_text = str(url).strip()
    parsed_url = (
        url_text
        if url_text.startswith(("http://", "https://"))
        else f"https://{url_text}"
    )
    parsed = urlparse(parsed_url)
    return url_text, parsed.netloc, parsed.path, parsed.query, parsed.fragment


def extract_url_features(url: str) -> dict[str, float | int]:
    url_text, domain, path, query, fragment = _normalize_url(url)

    url_length = len(url_text)
    digits = sum(char.isdigit() for char in url_text)
    letters = sum(char.isalpha() for char in url_text)
    lowercase = sum(char.islower() for char in url_text)
    uppercase = sum(char.isupper() for char in url_text)
    special_chars = sum(not char.isalnum() for char in url_text)

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

    return {
        "dots": url_text.count("."),
        "at": url_text.count("@"),
        "equals": url_text.count("="),
        "slashes": url_text.count("/"),
        "hyphens": url_text.count("-"),
        "colons": url_text.count(":"),
        "question_marks": url_text.count("?"),
        "digits": digits,
        "underscore": url_text.count("_"),
        "tilde": url_text.count("~"),
        "percent": url_text.count("%"),
        "lowercase": lowercase,
        "uppercase": uppercase,
        "upper_to_lower_ratio": uppercase / lowercase if lowercase else 0,
        "is_https": int(url_text.startswith("https://")),
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


def build_feature_frame(
    urls: list[str] | tuple[str, ...], feature_names: list[str] | None = None
) -> pd.DataFrame:
    feature_names = feature_names or FEATURE_NAMES
    feature_dicts = [extract_url_features(url) for url in urls]
    frame = pd.DataFrame(feature_dicts)
    return frame.reindex(columns=feature_names, fill_value=0)


def build_single_feature_frame(
    url: str, feature_names: list[str] | None = None
) -> pd.DataFrame:
    return build_feature_frame([url], feature_names)
