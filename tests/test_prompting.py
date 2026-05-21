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


def test_sm3_adapted_prompt_files_are_renderable_and_contract_aligned():
    prompt_files = {
        "sql": Path("prompts/sm3_adapted_sql_zero_shot.txt"),
        "sparql": Path("prompts/sm3_adapted_sparql_zero_shot.txt"),
        "cypher": Path("prompts/sm3_adapted_cypher_zero_shot.txt"),
    }
    rendered_texts = {}

    for language, path in prompt_files.items():
        text = path.read_text(encoding="utf-8")
        assert "{schema}" in text
        assert "{question}" in text
        assert "[Examples]" not in text

        template = PromptTemplate.from_path(language, path)
        rendered = render_prompt(
            template,
            schema="schema-placeholder",
            question="How many records are there?",
        )
        assert "schema-placeholder" in rendered.text
        assert "How many records are there?" in rendered.text
        rendered_texts[language] = rendered.text

    sql_prompt = prompt_files["sql"].read_text(encoding="utf-8")
    assert "SQLite SQL" in sql_prompt
    assert "Postgres" not in sql_prompt

    sparql_prompt = prompt_files["sparql"].read_text(encoding="utf-8")
    assert "Do not output PREFIX declarations" in sparql_prompt
    assert "RDF4J prepends repository prefixes automatically" in sparql_prompt
    assert ":class_name" in sparql_prompt
    assert "singer:name" in sparql_prompt
    assert "SELECT (COUNT(*) AS ?count) WHERE {{ ... }}" in sparql_prompt
    assert "SELECT (COUNT(*) AS ?count) WHERE { ... }" in rendered_texts["sparql"]
    for synthea_only_prefix in ("snomed", "uuid", "cvx", "udi", "loinc"):
        assert synthea_only_prefix not in sparql_prompt

    cypher_prompt = prompt_files["cypher"].read_text(encoding="utf-8")
    assert "single syntactically correct Neo4j Cypher MATCH query" in cypher_prompt
