from __future__ import annotations

import re

QUERY_STARTS = {
    "sql": ["WITH", "SELECT"],
    "sparql": ["SELECT", "ASK", "CONSTRUCT", "DESCRIBE"],
    "cypher": ["MATCH", "WITH", "CALL"],
}


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"```[a-zA-Z0-9_-]*\s*", "", text)
    return text.replace("```", "")


def _extract_from_first_keyword(text: str, language: str) -> str:
    upper_text = text.upper()
    positions = [
        upper_text.find(keyword)
        for keyword in QUERY_STARTS[language]
        if upper_text.find(keyword) >= 0
    ]
    if not positions:
        return text
    return text[min(positions) :]


def postprocess_completion(raw_completion: str, language: str) -> str:
    if language not in QUERY_STARTS:
        raise ValueError(f"Unsupported language: {language}")

    text = _strip_code_fences(_strip_think_blocks(raw_completion)).strip()
    if text.lower().startswith("no answer possible"):
        return ""
    text = _extract_from_first_keyword(text, language).strip()
    text = text.strip("`'\" \n\t")
    return re.sub(r"\s+", " ", text).strip()
