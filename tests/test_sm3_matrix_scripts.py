import os
import subprocess
from pathlib import Path


def _write_fake_command(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _fake_matrix_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_file = tmp_path / "commands.log"
    _write_fake_command(
        bin_dir / "python",
        'printf "python %s\\n" "$*" >> "${COMMAND_LOG}"\n'
        'case "$*" in\n'
        '  "- main") printf "qwen3_4b_instruct_2507\\n" ;;\n'
        '  "- main qwen3_4b_instruct_2507") printf "Qwen/Qwen3-4B-Instruct-2507\\n" ;;\n'
        '  "- openai") printf "gpt54_mini_20260317\\n" ;;\n'
        "esac\n",
    )
    _write_fake_command(
        bin_dir / "spider4ssc-zeroshot",
        'printf "spider4ssc-zeroshot %s\\n" "$*" >> "${COMMAND_LOG}"\n',
    )
    env = {
        **os.environ,
        "COMMAND_LOG": str(log_file),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    return env, log_file


def test_run_sm3_matrix_runs_all_languages_with_isolated_config(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env, log_file = _fake_matrix_env(tmp_path)

    subprocess.run(
        ["bash", "scripts/run_sm3_matrix.sh", "main", "dev", "7"],
        check=True,
        cwd=repo_root,
        env=env,
        input="\n",
        text=True,
        capture_output=True,
    )

    commands = log_file.read_text(encoding="utf-8")
    assert (
        "python scripts/wait_for_vllm.py "
        "--expected-model Qwen/Qwen3-4B-Instruct-2507"
    ) in commands
    for language in ("sql", "sparql", "cypher"):
        assert (
            f"spider4ssc-zeroshot generate qwen3_4b_instruct_2507 {language} "
            "--config configs/experiment_sm3_adapted.yaml "
            "--model-group main --split dev --schema-mode strict --limit 7"
        ) in commands
        assert (
            f"spider4ssc-zeroshot evaluate qwen3_4b_instruct_2507 {language} "
            "--config configs/experiment_sm3_adapted.yaml "
            "--split dev --schema-mode strict"
        ) in commands
    assert (
        "spider4ssc-zeroshot report --split dev "
        "--runs-dir runs/sm3_adapted/dev --output-dir reports/sm3_adapted"
    ) in commands


def test_run_sm3_dev_and_test_wrappers_select_split(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env, log_file = _fake_matrix_env(tmp_path)

    subprocess.run(
        ["bash", "scripts/run_sm3_dev.sh", "main", "3"],
        check=True,
        cwd=repo_root,
        env=env,
        input="\n",
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["bash", "scripts/run_sm3_test.sh", "main", "2"],
        check=True,
        cwd=repo_root,
        env=env,
        input="\n",
        text=True,
        capture_output=True,
    )

    commands = log_file.read_text(encoding="utf-8")
    assert "--split dev --schema-mode strict --limit 3" in commands
    assert "--runs-dir runs/sm3_adapted/dev --output-dir reports/sm3_adapted" in commands
    assert "--split test --schema-mode strict --limit 2" in commands
    assert "--runs-dir runs/sm3_adapted/test --output-dir reports/sm3_adapted" in commands


def test_run_sm3_openai_requires_api_key(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env, _ = _fake_matrix_env(tmp_path)
    env.pop("OPENAI_API_KEY", None)

    result = subprocess.run(
        ["bash", str(repo_root / "scripts/run_sm3_openai.sh"), "dev", "5"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "OPENAI_API_KEY is required" in result.stderr


def test_run_sm3_openai_runs_all_languages_without_vllm(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env, log_file = _fake_matrix_env(tmp_path)
    env["OPENAI_API_KEY"] = "sk-test"

    result = subprocess.run(
        ["bash", "scripts/run_sm3_openai.sh", "dev", "5"],
        check=True,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )

    commands = log_file.read_text(encoding="utf-8")
    assert "wait_for_vllm.py" not in commands
    assert "Start vLLM" not in result.stdout
    for language in ("sparql", "sql", "cypher"):
        assert (
            f"spider4ssc-zeroshot generate gpt54_mini_20260317 {language} "
            "--config configs/experiment_sm3_openai.yaml "
            "--model-group openai --split dev --schema-mode strict --limit 5"
        ) in commands
        assert (
            f"spider4ssc-zeroshot evaluate gpt54_mini_20260317 {language} "
            "--config configs/experiment_sm3_openai.yaml "
            "--split dev --schema-mode strict"
        ) in commands
    assert (
        "spider4ssc-zeroshot report --split dev "
        "--runs-dir runs/sm3_adapted/dev --output-dir reports/sm3_adapted"
    ) in commands


def test_run_sm3_openai_loads_api_key_from_dotenv(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    env, log_file = _fake_matrix_env(tmp_path)
    env.pop("OPENAI_API_KEY", None)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")

    subprocess.run(
        ["bash", str(repo_root / "scripts/run_sm3_openai.sh"), "dev", "1"],
        check=True,
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )

    commands = log_file.read_text(encoding="utf-8")
    assert "spider4ssc-zeroshot generate gpt54_mini_20260317 sparql" in commands
