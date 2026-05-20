import asyncio
import json
from types import SimpleNamespace

from spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq import serve_rdf4j_graphs


def test_rdf4j_test_split_loads_ttls_from_database_test(monkeypatch, tmp_path):
    dataset = tmp_path / "Spider4SSC"
    ttl_dir = dataset / "database_test" / "tiny_school"
    ttl_dir.mkdir(parents=True)
    (ttl_dir / "tiny_school.ttl").write_text("@prefix : <urn:test/> .\n", encoding="utf-8")
    (dataset / "test.json").write_text(
        json.dumps([{"db_id": "tiny_school"}]),
        encoding="utf-8",
    )
    fake_connector = _FakeRDF4jConnector()
    monkeypatch.setattr(
        serve_rdf4j_graphs,
        "RDF4jConnector",
        lambda: fake_connector,
    )

    asyncio.run(
        serve_rdf4j_graphs.main(
            SimpleNamespace(dataset_folder=dataset, split="test", db_subfolder=None)
        )
    )

    assert fake_connector.loaded_ttl_paths == [ttl_dir / "tiny_school.ttl"]


class _FakeRDF4jConnector:
    def __init__(self) -> None:
        self.loaded_ttl_paths = []

    async def create_repository(self, repository_id):
        pass

    async def add_data_to_named_graph(self, repository_id, graph_uri, ttl_file_path):
        self.loaded_ttl_paths.append(ttl_file_path)
