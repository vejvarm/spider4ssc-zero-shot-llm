from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptTemplate:
    language: str
    source_path: Path
    template: str

    @classmethod
    def from_path(cls, language: str, path: Path) -> PromptTemplate:
        return cls(language=language, source_path=path, template=path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RenderedPrompt:
    language: str
    text: str
    character_count: int


def render_prompt(template: PromptTemplate, *, schema: str, question: str) -> RenderedPrompt:
    text = template.template.format(schema=schema, question=question)
    return RenderedPrompt(
        language=template.language,
        text=text,
        character_count=len(text),
    )
