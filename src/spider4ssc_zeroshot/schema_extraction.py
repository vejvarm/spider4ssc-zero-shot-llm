from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from spider4ssc_zeroshot.data import load_split, split_db_dir_name, split_db_root
from spider4ssc_zeroshot.vendor.ut5_ssc.seq2seq.utils.neo4j_schema_extractor import (
    Neo4jSchemaExtractor,
)
from spider4ssc_zeroshot.vendor.ut5_ssc.third_party.test_suite.neo4j_connector import (
    Neo4jConnector,
)


def _default_import_subfolder(db_subfolder: str) -> str:
    return f"import/Spider4SSC/{db_subfolder}"


async def _extract_neo4j_schemas_async(
    dataset_root: Path,
    *,
    split: str,
    neo4j_root: Path,
    test_file: str,
    test_db_dir: str,
    import_subfolder: str | None,
    overwrite: bool,
    wipe: bool,
) -> dict[str, Any]:
    db_subfolder = split_db_dir_name(split, test_db_dir=test_db_dir)
    effective_import_subfolder = import_subfolder or _default_import_subfolder(
        db_subfolder
    )
    db_root = split_db_root(dataset_root, split, test_db_dir=test_db_dir)
    db_ids = sorted(
        {example.db_id for example in load_split(dataset_root, split, test_file=test_file)}
    )

    connector = Neo4jConnector(
        username="neo4j",
        password="secretserver",
        neo4j_root=neo4j_root,
        database_subfolder=db_subfolder,
        host_db_root=dataset_root,
        import_subfolder=effective_import_subfolder,
    )
    extractor = Neo4jSchemaExtractor(db_root=db_root)
    extracted: list[str] = []
    skipped_existing: list[str] = []
    failed: dict[str, str] = {}
    try:
        if wipe:
            await connector.wipe_databases()
        for db_id in db_ids:
            schema_path = db_root / db_id / f"{db_id}.neo4j-schema.json"
            if schema_path.exists() and not overwrite:
                skipped_existing.append(db_id)
                continue
            try:
                await connector.create_database(db_id)
                await connector.use_database(db_id)
                prefixes = connector.extract_prefixes_from_ttl(db_id)
                await connector.init_database(prefixes, db_id)
                extractor.extract_schema(
                    db_id,
                    restructure=True,
                    dump=True,
                    overwrite=overwrite,
                    init_db=False,
                )
                extracted.append(db_id)
            except Exception as error:  # pragma: no cover - depends on live Neo4j state
                failed[db_id] = repr(error)
    finally:
        extractor.close()
        await connector.close()

    return {
        "dataset_root": str(dataset_root),
        "split": split,
        "database_root": str(db_root),
        "neo4j_root": str(neo4j_root),
        "database_subfolder": db_subfolder,
        "import_subfolder": effective_import_subfolder,
        "overwrite": overwrite,
        "wipe": wipe,
        "n_databases": len(db_ids),
        "extracted": extracted,
        "skipped_existing": skipped_existing,
        "failed": failed,
    }


def extract_neo4j_schemas(
    dataset_root: Path,
    *,
    split: str,
    neo4j_root: Path,
    test_file: str = "test.json",
    test_db_dir: str = "database_test",
    import_subfolder: str | None = None,
    overwrite: bool = False,
    wipe: bool = True,
) -> dict[str, Any]:
    return asyncio.run(
        _extract_neo4j_schemas_async(
            dataset_root.resolve(),
            split=split,
            neo4j_root=neo4j_root.resolve(),
            test_file=test_file,
            test_db_dir=test_db_dir,
            import_subfolder=import_subfolder,
            overwrite=overwrite,
            wipe=wipe,
        )
    )
