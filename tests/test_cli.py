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
        "  output_root: runs/test\n"
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
        "  output_root: runs/test\n"
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
    assert captured["wait_model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert captured["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert captured["output_file"] == (
        tmp_path / "runs" / "qwen3_4b_instruct_2507" / "sql" / "predictions.jsonl"
    )


def test_cli_exposes_reproducibility_commands():
    for command in ["prepare-data", "generate", "evaluate", "report"]:
        result = CliRunner().invoke(app, [command, "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output
