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


def test_postprocess_no_answer_sentence_returns_empty_string():
    raw_completion = "No answer possible based on given input."

    assert postprocess_completion(raw_completion, "sql") == ""


def test_postprocess_unsupported_language_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported language"):
        postprocess_completion("SELECT 1", "gremlin")
