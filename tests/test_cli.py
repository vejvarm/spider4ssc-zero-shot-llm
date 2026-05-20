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
