import os
import subprocess
from pathlib import Path


def test_setup_env_vars_can_be_sourced_for_model_cache_paths(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    cache_root = tmp_path / "models"

    result = subprocess.run(
        [
            "bash",
            "-lc",
            (
                "unset HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE VLLM_DOWNLOAD_DIR; "
                "source scripts/setup_env_vars.sh >/dev/null; "
                'printf "HF_HOME=%s\\n" "$HF_HOME"; '
                'printf "HF_HUB_CACHE=%s\\n" "$HF_HUB_CACHE"; '
                'printf "TRANSFORMERS_CACHE=%s\\n" "$TRANSFORMERS_CACHE"; '
                'printf "VLLM_DOWNLOAD_DIR=%s\\n" "$VLLM_DOWNLOAD_DIR"'
            ),
        ],
        check=True,
        cwd=repo_root,
        env={**os.environ, "SPIDER4SSC_MODEL_CACHE_ROOT": str(cache_root)},
        text=True,
        capture_output=True,
    )

    assert f"HF_HOME={cache_root}/hf" in result.stdout
    assert f"HF_HUB_CACHE={cache_root}/hf/hub" in result.stdout
    assert f"TRANSFORMERS_CACHE={cache_root}/hf/transformers" in result.stdout
    assert f"VLLM_DOWNLOAD_DIR={cache_root}/vllm" in result.stdout
    assert (cache_root / "hf" / "hub").is_dir()
    assert (cache_root / "hf" / "transformers").is_dir()
    assert (cache_root / "vllm").is_dir()


def test_setup_env_vars_can_wrap_a_command(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    cache_root = tmp_path / "models"

    result = subprocess.run(
        [
            "bash",
            "scripts/setup_env_vars.sh",
            "bash",
            "-c",
            'printf "%s\\n%s\\n" "$HF_HOME" "$VLLM_DOWNLOAD_DIR"',
        ],
        check=True,
        cwd=repo_root,
        env={
            **os.environ,
            "SPIDER4SSC_MODEL_CACHE_ROOT": str(cache_root),
            "HF_HOME": "/main-disk/hf",
            "HF_HUB_CACHE": "/main-disk/hub",
            "TRANSFORMERS_CACHE": "/main-disk/transformers",
            "VLLM_DOWNLOAD_DIR": "/main-disk/vllm",
        },
        text=True,
        capture_output=True,
    )

    assert result.stdout.splitlines() == [
        f"{cache_root}/hf",
        f"{cache_root}/vllm",
    ]
