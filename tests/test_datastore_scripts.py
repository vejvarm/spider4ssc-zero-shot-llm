import asyncio
import json
from types import SimpleNamespace

from spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq import (
    serve_neo4j_graphs,
    serve_rdf4j_graphs,
)


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


def test_neo4j_test_split_uses_separate_host_and_import_subfolders(
    monkeypatch,
    tmp_path,
):
    dataset = tmp_path / "Spider4SSC"
    ttl_dir = dataset / "database_test" / "tiny_school"
    ttl_dir.mkdir(parents=True)
    (ttl_dir / "tiny_school.ttl").write_text("@prefix : <urn:test/> .\n", encoding="utf-8")
    (dataset / "test.json").write_text(
        json.dumps([{"db_id": "tiny_school"}]),
        encoding="utf-8",
    )
    fake_connector = _FakeNeo4jConnector()

    def fake_connector_factory(**kwargs):
        fake_connector.kwargs = kwargs
        return fake_connector

    monkeypatch.setattr(
        serve_neo4j_graphs,
        "Neo4jConnector",
        fake_connector_factory,
    )

    asyncio.run(
        serve_neo4j_graphs.main(
            SimpleNamespace(
                dataset_folder=dataset,
                split="test",
                neo4j_root=tmp_path / "neo4j-root",
                db_subfolder=None,
                import_subfolder="import/Spider4SSC/database_test",
                wipe=True,
            )
        )
    )

    assert fake_connector.kwargs["database_subfolder"] == "database_test"
    assert fake_connector.kwargs["host_db_root"] == dataset
    assert fake_connector.kwargs["import_subfolder"] == "import/Spider4SSC/database_test"
    assert fake_connector.prefix_names == ["tiny_school"]
    assert fake_connector.initialized == [("tiny_school", {"ROOT": "urn:test/"})]


class _FakeNeo4jConnector:
    def __init__(self) -> None:
        self.kwargs = {}
        self.prefix_names = []
        self.initialized = []

    async def wipe_databases(self):
        pass

    async def create_database(self, database_name):
        pass

    async def use_database(self, database_name):
        pass

    def extract_prefixes_from_ttl(self, db_name):
        self.prefix_names.append(db_name)
        return {"ROOT": "urn:test/"}

    async def init_database(self, prefixes, db_name):
        self.initialized.append((db_name, prefixes))

    async def close(self):
        pass
