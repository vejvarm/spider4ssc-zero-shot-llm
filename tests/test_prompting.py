from pathlib import Path

from spider4ssc_zeroshot.prompting import PromptTemplate, render_prompt


def test_render_prompt_substitutes_schema_question_and_records_metadata(tmp_path: Path):
    template_path = tmp_path / "sql_prompt.txt"
    template_path.write_text("Schema:\n{schema}\nQuestion: {question}", encoding="utf-8")

    template = PromptTemplate.from_path("sql", template_path)
    rendered = render_prompt(
        template,
        schema="CREATE TABLE student(id INTEGER);",
        question="How many students are there?",
    )

    assert template.language == "sql"
    assert template.source_path == template_path
    assert rendered.language == "sql"
    assert rendered.text == (
        "Schema:\nCREATE TABLE student(id INTEGER);\n"
        "Question: How many students are there?"
    )
    assert rendered.character_count == len(rendered.text)
