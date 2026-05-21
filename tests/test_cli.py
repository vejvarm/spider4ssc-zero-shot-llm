import json

from typer.testing import CliRunner

from spider4ssc_zeroshot import cli
from spider4ssc_zeroshot.cli import app


def test_prepare_data_copies_source_and_writes_manifest(tmp_path):
    output = tmp_path / "Spider4SSC"

    result = CliRunner().invoke(
        app,
        [
            "prepare-data",
            "--source",
            "tests/fixtures/tiny_spider4ssc",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert (output / "test.json").exists()
    assert (tmp_path / "Spider4SSC.manifest.json").exists()
    assert f"Prepared Spider4SSC at {output}" in result.output


def test_prepare_data_uses_config_local_path_when_output_is_omitted(tmp_path):
    output = tmp_path / "configured" / "Spider4SSC"
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "dataset:\n"
        "  name: Spider4SSC\n"
        "  url: https://example.org/Spider4SSC.tgz\n"
        f"  local_path: {output.as_posix()}\n"
        "  split: test\n"
        "  test_file: test.json\n"
        "  test_db_dir: database_test\n"
        "  archive_sha256: null\n"
        "experiment:\n"
        "  schema_serialization: compact\n"
        "  languages:\n"
        "    - sql\n"
        "  prompt_files:\n"
        "    sql: prompts/sql_zero_shot.txt\n"
        "  schema_mode: strict\n"
        "  output_root: runs\n"
        "  report_dir: reports\n"
        "decoding: {}\n"
        "endpoint: {}\n"
        "reproducibility:\n"
        "  forbid_prompt_change_after_full_run: true\n"
        "  record_full_prompt: true\n"
        "  record_raw_completion: true\n"
        "  record_model_revision: true\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "prepare-data",
            "--source",
            "tests/fixtures/tiny_spider4ssc",
            "--config",
            str(config),
        ],
    )

    assert result.exit_code == 0
    assert (output / "test.json").exists()
    assert (output.parent / "Spider4SSC.manifest.json").exists()
    assert f"Prepared Spider4SSC at {output}" in result.output


def test_prepare_data_without_source_requires_remote_archive_checksum(tmp_path):
    output = tmp_path / "configured" / "Spider4SSC"
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "dataset:\n"
        "  name: Spider4SSC\n"
        "  url: https://example.org/Spider4SSC.tgz\n"
        f"  local_path: {output.as_posix()}\n"
        "  split: test\n"
        "  test_file: test.json\n"
        "  test_db_dir: database_test\n"
        "  archive_sha256: null\n"
        "experiment:\n"
        "  schema_serialization: compact\n"
        "  languages:\n"
        "    - sql\n"
        "  prompt_files:\n"
        "    sql: prompts/sql_zero_shot.txt\n"
        "  schema_mode: strict\n"
        "  output_root: runs\n"
        "  report_dir: reports\n"
        "decoding: {}\n"
        "endpoint: {}\n"
        "reproducibility:\n"
        "  forbid_prompt_change_after_full_run: true\n"
        "  record_full_prompt: true\n"
        "  record_raw_completion: true\n"
        "  record_model_revision: true\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["prepare-data", "--config", str(config)])

    assert result.exit_code != 0
    assert "archive_sha256 is required" in result.output
    assert "Traceback" not in result.output


def test_generate_uses_configured_test_split_file_before_vllm(monkeypatch, tmp_path):
    dataset_root = tmp_path / "Spider4SSC"
    dataset_root.mkdir()
    (dataset_root / "custom_test.json").write_text(
        json.dumps(
            [
                {
                    "db_id": "tiny_school",
                    "question": "How many students are there?",
                    "sql": "SELECT count(*) FROM student",
                }
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "dataset:\n"
        "  name: Spider4SSC\n"
        "  url: https://example.org/Spider4SSC.tgz\n"
        f"  local_path: {dataset_root.as_posix()}\n"
        "  split: test\n"
        "  test_file: custom_test.json\n"
        "  test_db_dir: custom_database\n"
        "  archive_sha256: null\n"
        "experiment:\n"
        "  schema_serialization: compact\n"
        "  languages:\n"
        "    - sql\n"
        "  prompt_files:\n"
        "    sql: prompts/sql_zero_shot.txt\n"
        "  schema_mode: strict\n"
        f"  output_root: {(tmp_path / 'runs').as_posix()}\n"
        f"  report_dir: {(tmp_path / 'reports').as_posix()}\n"
        "decoding: {}\n"
        "endpoint: {}\n"
        "reproducibility:\n"
        "  forbid_prompt_change_after_full_run: true\n"
        "  record_full_prompt: true\n"
        "  record_raw_completion: true\n"
        "  record_model_revision: true\n",
        encoding="utf-8",
    )
    captured = {}

    class FakeClient:
        def __init__(self, config):
            self.config = config

        def wait_until_ready(self, model_id):
            captured["wait_model_id"] = model_id

    def fake_run_generation(requests, client, model_id, decoding, output_file):
        captured["requests"] = requests
        captured["model_id"] = model_id
        captured["output_file"] = output_file

    monkeypatch.setattr(cli, "VllmClient", FakeClient)
    monkeypatch.setattr(cli, "serialize_example_schema", lambda *args, **kwargs: "schema")
    monkeypatch.setattr(
        cli,
        "schema_provenance_for_example",
        lambda *args, **kwargs: "sqlite-schema-dump",
    )
    monkeypatch.setattr(cli, "run_generation", fake_run_generation)

    result = CliRunner().invoke(
        app,
        [
            "generate",
            "qwen3_4b_instruct_2507",
            "sql",
            "--config",
            str(config),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert captured["requests"][0].db_id == "tiny_school"
    assert captured["requests"][0].gold_sql == "SELECT count(*) FROM student"
    assert captured["requests"][0].gold_query == "SELECT count(*) FROM student"
    assert captured["requests"][0].schema_mode == "strict"
    assert captured["requests"][0].schema_provenance == "sqlite-schema-dump"
    assert captured["wait_model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert captured["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert captured["output_file"] == (
        tmp_path
        / "runs"
        / "test"
        / "qwen3_4b_instruct_2507"
        / "sql"
        / "predictions.jsonl"
    )


def test_generate_selects_openai_client_and_records_provider(monkeypatch, tmp_path):
    dataset_root = tmp_path / "Spider4SSC"
    dataset_root.mkdir()
    (dataset_root / "test.json").write_text(
        json.dumps(
            [
                {
                    "db_id": "tiny_school",
                    "question": "How many students are there?",
                    "sql": "SELECT count(*) FROM student",
                }
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "experiment_openai.yaml"
    config.write_text(
        "dataset:\n"
        "  name: Spider4SSC\n"
        "  url: https://example.org/Spider4SSC.tgz\n"
        f"  local_path: {dataset_root.as_posix()}\n"
        "  split: test\n"
        "  test_file: test.json\n"
        "  test_db_dir: database_test\n"
        "  archive_sha256: null\n"
        "experiment:\n"
        "  schema_serialization: compact\n"
        "  languages:\n"
        "    - sql\n"
        "  prompt_files:\n"
        "    sql: prompts/sql_zero_shot.txt\n"
        "  schema_mode: strict\n"
        f"  output_root: {(tmp_path / 'runs' / 'sm3_adapted').as_posix()}\n"
        f"  report_dir: {(tmp_path / 'reports' / 'sm3_adapted').as_posix()}\n"
        "decoding:\n"
        "  reasoning_effort: none\n"
        "endpoint:\n"
        "  base_url: https://api.openai.com/v1\n"
        "  api_key_env: OPENAI_API_KEY\n"
        "reproducibility:\n"
        "  forbid_prompt_change_after_full_run: true\n"
        "  record_full_prompt: true\n"
        "  record_raw_completion: true\n"
        "  record_model_revision: true\n",
        encoding="utf-8",
    )
    models = tmp_path / "models.yaml"
    models.write_text(
        "openai:\n"
        "  gpt54_mini_20260317:\n"
        "    provider: openai\n"
        "    model_id: gpt-5.4-mini-2026-03-17\n"
        "    family: gpt-5.4\n"
        "    size_label: mini\n",
        encoding="utf-8",
    )
    captured = {}

    class FakeOpenAIClient:
        def __init__(self, config):
            captured["client_config"] = config

        def wait_until_ready(self, model_id):
            captured["wait_model_id"] = model_id

    def fake_run_generation(requests, client, model_id, decoding, output_file):
        captured["requests"] = requests
        captured["client"] = client
        captured["model_id"] = model_id
        captured["decoding"] = decoding
        captured["output_file"] = output_file

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "OpenAIChatClient", FakeOpenAIClient)
    monkeypatch.setattr(cli, "VllmClient", lambda config: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(cli, "serialize_example_schema", lambda *args, **kwargs: "schema")
    monkeypatch.setattr(
        cli,
        "schema_provenance_for_example",
        lambda *args, **kwargs: "sqlite-schema-dump",
    )
    monkeypatch.setattr(cli, "run_generation", fake_run_generation)

    result = CliRunner().invoke(
        app,
        [
            "generate",
            "gpt54_mini_20260317",
            "sql",
            "--config",
            str(config),
            "--models",
            str(models),
            "--model-group",
            "openai",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert captured["client_config"].base_url == "https://api.openai.com/v1"
    assert captured["client_config"].api_key == "sk-test"
    assert captured["wait_model_id"] == "gpt-5.4-mini-2026-03-17"
    assert captured["model_id"] == "gpt-5.4-mini-2026-03-17"
    assert captured["decoding"]["reasoning_effort"] == "none"
    assert captured["requests"][0].model_provider == "openai"
    assert captured["output_file"] == (
        tmp_path
        / "runs"
        / "sm3_adapted"
        / "test"
        / "gpt54_mini_20260317"
        / "sql"
        / "predictions.jsonl"
    )


def test_generate_loads_openai_api_key_from_dotenv(monkeypatch, tmp_path):
    dataset_root = tmp_path / "Spider4SSC"
    dataset_root.mkdir()
    (dataset_root / "test.json").write_text(
        json.dumps(
            [
                {
                    "db_id": "tiny_school",
                    "question": "How many students are there?",
                    "sql": "SELECT count(*) FROM student",
                }
            ]
        ),
        encoding="utf-8",
    )
    prompt_file = tmp_path / "sql_prompt.txt"
    prompt_file.write_text("{schema}\n{question}\n", encoding="utf-8")
    config = tmp_path / "experiment_openai.yaml"
    config.write_text(
        "dataset:\n"
        "  name: Spider4SSC\n"
        "  url: https://example.org/Spider4SSC.tgz\n"
        f"  local_path: {dataset_root.as_posix()}\n"
        "  split: test\n"
        "  test_file: test.json\n"
        "  test_db_dir: database_test\n"
        "  archive_sha256: null\n"
        "experiment:\n"
        "  schema_serialization: compact\n"
        "  languages:\n"
        "    - sql\n"
        "  prompt_files:\n"
        f"    sql: {prompt_file.as_posix()}\n"
        "  schema_mode: strict\n"
        f"  output_root: {(tmp_path / 'runs' / 'sm3_adapted').as_posix()}\n"
        f"  report_dir: {(tmp_path / 'reports' / 'sm3_adapted').as_posix()}\n"
        "decoding:\n"
        "  reasoning_effort: none\n"
        "endpoint:\n"
        "  base_url: https://api.openai.com/v1\n"
        "  api_key_env: OPENAI_API_KEY\n"
        "reproducibility:\n"
        "  forbid_prompt_change_after_full_run: true\n"
        "  record_full_prompt: true\n"
        "  record_raw_completion: true\n"
        "  record_model_revision: true\n",
        encoding="utf-8",
    )
    models = tmp_path / "models.yaml"
    models.write_text(
        "openai:\n"
        "  gpt54_mini_20260317:\n"
        "    provider: openai\n"
        "    model_id: gpt-5.4-mini-2026-03-17\n"
        "    family: gpt-5.4\n"
        "    size_label: mini\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")
    captured = {}

    class FakeOpenAIClient:
        def __init__(self, config):
            captured["client_config"] = config

        def wait_until_ready(self, model_id):
            pass

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "OpenAIChatClient", FakeOpenAIClient)
    monkeypatch.setattr(cli, "serialize_example_schema", lambda *args, **kwargs: "schema")
    monkeypatch.setattr(
        cli,
        "schema_provenance_for_example",
        lambda *args, **kwargs: "sqlite-schema-dump",
    )
    monkeypatch.setattr(cli, "run_generation", lambda *args, **kwargs: None)

    result = CliRunner().invoke(
        app,
        [
            "generate",
            "gpt54_mini_20260317",
            "sql",
            "--config",
            str(config),
            "--models",
            str(models),
            "--model-group",
            "openai",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert captured["client_config"].api_key == "sk-from-dotenv"


def test_validate_pipeline_reports_errors_without_traceback(tmp_path):
    dataset_root = tmp_path / "Spider4SSC"
    db_dir = dataset_root / "database_test" / "tiny_school"
    db_dir.mkdir(parents=True)
    (dataset_root / "test.json").write_text(
        json.dumps(
            [
                {
                    "db_id": "tiny_school",
                    "question": "How many students are there?",
                    "sql": "SELECT count(*) FROM student",
                }
            ]
        ),
        encoding="utf-8",
    )
    (db_dir / "tiny_school.sqlite").write_text("sqlite", encoding="utf-8")
    (db_dir / "tiny_school.ttl").write_text("@prefix : <urn:test/> .\n", encoding="utf-8")
    (db_dir / "tiny_school.rdf-schema.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "dataset:\n"
        "  name: Spider4SSC\n"
        "  url: https://example.org/Spider4SSC.tgz\n"
        f"  local_path: {dataset_root.as_posix()}\n"
        "  split: test\n"
        "  test_file: test.json\n"
        "  test_db_dir: database_test\n"
        "  archive_sha256: null\n"
        "experiment:\n"
        "  schema_serialization: compact\n"
        "  schema_mode: strict\n"
        "  languages:\n"
        "    - sql\n"
        "  prompt_files:\n"
        "    sql: prompts/sql_zero_shot.txt\n"
        "  output_root: runs\n"
        "  report_dir: reports\n"
        "decoding: {}\n"
        "endpoint: {}\n"
        "reproducibility:\n"
        "  forbid_prompt_change_after_full_run: true\n"
        "  record_full_prompt: true\n"
        "  record_raw_completion: true\n"
        "  record_model_revision: true\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "validate-pipeline",
            "--config",
            str(config),
            "--no-enforce-expected-counts",
        ],
    )

    assert result.exit_code == 1
    assert "missing strict Neo4j schema" in result.output
    assert "Traceback" not in result.output


def test_cli_exposes_reproducibility_commands():
    for command in [
        "prepare-data",
        "generate",
        "evaluate",
        "report",
        "validate-pipeline",
        "extract-neo4j-schemas",
    ]:
        result = CliRunner().invoke(app, [command, "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output
