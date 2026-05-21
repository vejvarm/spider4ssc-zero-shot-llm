import os
import subprocess
from pathlib import Path


def _write_fake_command(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _fake_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_file = tmp_path / "commands.log"
    dataset_root = tmp_path / "Spider4SSC"
    (dataset_root / "database").mkdir(parents=True)
    (dataset_root / "database_test").mkdir(parents=True)

    _write_fake_command(
        bin_dir / "docker",
        'printf "docker %s\\n" "$*" >> "${COMMAND_LOG}"\n',
    )
    _write_fake_command(
        bin_dir / "curl",
        'printf "curl %s\\n" "$*" >> "${COMMAND_LOG}"\n',
    )
    _write_fake_command(
        bin_dir / "python",
        'printf "python %s\\n" "$*" >> "${COMMAND_LOG}"\n',
    )
    _write_fake_command(
        bin_dir / "spider4ssc-zeroshot",
        'printf "spider4ssc-zeroshot %s\\n" "$*" >> "${COMMAND_LOG}"\n',
    )

    env = {
        **os.environ,
        "COMMAND_LOG": str(log_file),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SPIDER4SSC_DATASET_ROOT": str(dataset_root),
    }
    return env, log_file, dataset_root


def test_serve_dev_restarts_and_loads_dev_datastores(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env, log_file, dataset_root = _fake_env(tmp_path)

    subprocess.run(
        ["bash", "scripts/serve_dev.sh"],
        check=True,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )

    commands = log_file.read_text(encoding="utf-8")
    assert "docker compose -f docker/compose.datastores.yml down -v" in commands
    assert "docker compose -f docker/compose.datastores.yml up -d" in commands
    assert (
        "python -m spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.serve_rdf4j_graphs "
        f"{dataset_root} --split dev --db-subfolder database"
    ) in commands
    assert (
        "python -m spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.serve_neo4j_graphs "
        f"{dataset_root} --split dev --neo4j-root docker/neo4j-root "
        "--db-subfolder database --import-subfolder import/Spider4SSC/database"
    ) in commands
    assert (
        "spider4ssc-zeroshot extract-neo4j-schemas --split dev "
        "--neo4j-root docker/neo4j-root "
        "--import-subfolder import/Spider4SSC/database --no-wipe"
    ) in commands
    assert (
        "spider4ssc-zeroshot validate-pipeline --split dev --schema-mode strict"
    ) in commands


def test_serve_test_restarts_and_loads_test_datastores(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env, log_file, dataset_root = _fake_env(tmp_path)

    subprocess.run(
        ["bash", "scripts/serve_test.sh"],
        check=True,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )

    commands = log_file.read_text(encoding="utf-8")
    assert (
        "python -m spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.serve_rdf4j_graphs "
        f"{dataset_root} --split test --db-subfolder database_test"
    ) in commands
    assert (
        "python -m spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.serve_neo4j_graphs "
        f"{dataset_root} --split test --neo4j-root docker/neo4j-root "
        "--db-subfolder database_test "
        "--import-subfolder import/Spider4SSC/database_test"
    ) in commands
    assert (
        "spider4ssc-zeroshot extract-neo4j-schemas --split test "
        "--neo4j-root docker/neo4j-root "
        "--import-subfolder import/Spider4SSC/database_test --no-wipe"
    ) in commands
    assert (
        "spider4ssc-zeroshot validate-pipeline --split test --schema-mode strict"
    ) in commands
