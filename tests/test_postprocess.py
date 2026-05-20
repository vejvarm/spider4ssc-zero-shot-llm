import pytest

from spider4ssc_zeroshot.postprocess import postprocess_completion


def test_postprocess_strips_think_block_and_code_fence():
    raw_completion = """<think>
I should explain.
</think>

```sql
SELECT COUNT(*) FROM student;
```
"""

    assert postprocess_completion(raw_completion, "sql") == "SELECT COUNT(*) FROM student;"


def test_postprocess_extracts_cypher_match():
    raw_completion = "The answer is:\nMATCH (s:Student) RETURN count(s)"

    assert postprocess_completion(raw_completion, "cypher") == "MATCH (s:Student) RETURN count(s)"


def test_postprocess_extracts_sparql_select():
    raw_completion = 'Here is the query:\n"SELECT ?s WHERE { ?s ?p ?o }"'

    assert postprocess_completion(raw_completion, "sparql") == "SELECT ?s WHERE { ?s ?p ?o }"


def test_postprocess_preserves_sparql_prefixes():
    raw_completion = (
        "Here is the query:\n"
        "PREFIX ex: <http://example.org/> SELECT ?s WHERE { ?s ex:name ?name }"
    )

    assert postprocess_completion(raw_completion, "sparql") == (
        "PREFIX ex: <http://example.org/> SELECT ?s WHERE { ?s ex:name ?name }"
    )


def test_postprocess_preserves_cypher_optional_match():
    raw_completion = "Query:\nOPTIONAL MATCH (s:Student) RETURN s.name"

    assert postprocess_completion(raw_completion, "cypher") == (
        "OPTIONAL MATCH (s:Student) RETURN s.name"
    )


def test_postprocess_no_answer_sentence_returns_empty_string():
    raw_completion = "No answer possible based on given input."

    assert postprocess_completion(raw_completion, "sql") == ""


def test_postprocess_no_answer_with_extra_query_returns_empty_string():
    raw_completion = "No answer possible based on given input. SELECT 1"

    assert postprocess_completion(raw_completion, "sql") == ""


def test_postprocess_rejects_prose_that_mentions_query_keyword():
    raw_completion = "I cannot answer this from the schema; no SELECT query is possible."

    assert postprocess_completion(raw_completion, "sql") == ""


def test_postprocess_unclosed_think_block_returns_empty_string():
    raw_completion = "<think>I should SELECT the table first\nSELECT COUNT(*) FROM student;"

    assert postprocess_completion(raw_completion, "sql") == ""


def test_postprocess_unsupported_language_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported language"):
        postprocess_completion("SELECT 1", "gremlin")
