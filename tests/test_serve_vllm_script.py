import os
import subprocess
import sys
from pathlib import Path


def test_serve_vllm_filters_preloaded_nccl(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    fake_vllm = tmp_path / "vllm"
    fake_vllm.write_text(
        "#!/usr/bin/env bash\n"
        'printf "LD_PRELOAD=%s\\n" "${LD_PRELOAD:-}"\n'
        'printf "ARGS=%s\\n" "$*"\n',
        encoding="utf-8",
    )
    fake_vllm.chmod(0o755)

    env = {
        **os.environ,
        "LD_PRELOAD": "/usr/local/lib/libmsamp_dist.so:/usr/lib/x86_64-linux-gnu/libnccl.so:",
        "PYTHON": sys.executable,
        "PYTHONPATH": "src",
        "VLLM_BIN": str(fake_vllm),
    }
    result = subprocess.run(
        ["bash", "scripts/serve_vllm.sh", "qwen3_4b_instruct_2507", "main"],
        check=True,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "LD_PRELOAD=/usr/local/lib/libmsamp_dist.so" in result.stdout
    assert "/usr/lib/x86_64-linux-gnu/libnccl.so" not in result.stdout
    assert "serve Qwen/Qwen3-4B-Instruct-2507" in result.stdout
    assert f"--download-dir {repo_root / 'models'}" in result.stdout
