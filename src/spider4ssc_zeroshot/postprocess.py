from __future__ import annotations

import re

_KEYWORDS_BY_LANGUAGE = {
    "sql": ("WITH", "SELECT"),
    "sparql": ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE"),
    "cypher": ("MATCH", "WITH", "CALL"),
}
_NO_ANSWER_PREFIX = "No answer possible based on given input"


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"```[a-zA-Z0-9_-]*\s*", "", text)
    return text.replace("```", "")


def _strip_wrapping_marks(text: str) -> str:
    return text.strip("`\"'").strip()


def _extract_from_first_keyword(text: str, keywords: tuple[str, ...]) -> str:
    keyword_pattern = "|".join(re.escape(keyword) for keyword in keywords)
    match = re.search(rf"\b({keyword_pattern})\b", text, flags=re.IGNORECASE)
    if match is None:
        return text
    return text[match.start() :]


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def postprocess_completion(raw_completion: str, language: str) -> str:
    if language not in _KEYWORDS_BY_LANGUAGE:
        raise ValueError(f"Unsupported language: {language}")

    text = _strip_think_blocks(raw_completion)
    text = _strip_code_fences(text)
    text = _strip_wrapping_marks(text.strip())
    text = _normalize_whitespace(text)

    if text.startswith(_NO_ANSWER_PREFIX):
        return ""

    text = _extract_from_first_keyword(text, _KEYWORDS_BY_LANGUAGE[language])
    text = _strip_wrapping_marks(text.strip())
    return _normalize_whitespace(text)
