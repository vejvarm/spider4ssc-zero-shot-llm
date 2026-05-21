import os

from spider4ssc_zeroshot.env import load_dotenv


def test_load_dotenv_reads_simple_key_values_without_overriding_existing_env(
    monkeypatch,
    tmp_path,
):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# local secrets\n"
        "OPENAI_API_KEY=sk-from-dotenv\n"
        "QUOTED_VALUE='quoted value'\n"
        "export EXPORTED_VALUE=exported\n"
        "EXISTING=value-from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)
    monkeypatch.delenv("EXPORTED_VALUE", raising=False)
    monkeypatch.setenv("EXISTING", "already-exported")

    load_dotenv(dotenv)

    assert os.environ["OPENAI_API_KEY"] == "sk-from-dotenv"
    assert os.environ["QUOTED_VALUE"] == "quoted value"
    assert os.environ["EXPORTED_VALUE"] == "exported"
    assert os.environ["EXISTING"] == "already-exported"
