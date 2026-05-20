# Vendored Spider4SSC Utilities

Copied on 2026-05-21 from `/home/vejvar-martin-nj/git/uT5-ssc` at commit
`1b84ba30065928c0ccb935bbe4c713d84895a5db`.

Vendored source paths:

- `third_party/test_suite/`
- `third_party/spider/`
- `seq2seq/utils/dataset.py`
- `seq2seq/utils/rdf_schema_extractor.py`
- `seq2seq/utils/neo4j_schema_extractor.py`
- `seq2seq/utils/cypher_identifier_mapping.py`
- `seq2seq/serve_rdf4j_graphs.py`
- `seq2seq/serve_neo4j_graphs.py`
- `seq2seq/metrics/spider/`
- `seq2seq/metrics/spidersparql/`
- `seq2seq/metrics/spidercypher/`

Local changes after copying:

- Import paths are rewritten from `third_party...` and `seq2seq...` to
  `spider4ssc_zeroshot.vendor.ut5_ssc...`.
- The debug `print(gold_sql, pred_query, pred_lang)` line in the copied SPARQL
  cross-language evaluator is removed.
- Nested `.git`, `__pycache__`, bytecode, upstream example databases, archives,
  and notebook/data examples are excluded from this reproducibility repository.
- Optional imports for upstream training-only dependencies are guarded so schema
  serialization can run without installing the full training stack.
- The RDF4J serving helper defaults to `database_test` for `--split test` and
  exposes `--db-subfolder` so test-set graph loading uses the same layout as SQL
  and Cypher evaluation.
- No metric behavior is intentionally changed.
