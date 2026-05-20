from typer.testing import CliRunner

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
