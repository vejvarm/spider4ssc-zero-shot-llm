import shutil
from pathlib import Path

VENDORED_ROOT = Path(__file__).resolve().parents[1] / "src/spider4ssc_zeroshot/vendor/ut5_ssc"


def _remove_vendored_bytecode() -> None:
    for cache_dir in VENDORED_ROOT.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    for bytecode_file in VENDORED_ROOT.rglob("*.pyc"):
        bytecode_file.unlink(missing_ok=True)


def pytest_sessionstart(session) -> None:
    _remove_vendored_bytecode()


def pytest_sessionfinish(session, exitstatus) -> None:
    _remove_vendored_bytecode()
