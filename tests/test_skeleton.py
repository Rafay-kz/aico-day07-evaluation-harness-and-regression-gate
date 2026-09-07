"""Prove the package layout is importable and CLIs parse arguments."""

from pathlib import Path

import aico
from aico.evals import day01, day02
from aico.platform import model_gateway
from aico.retrieval import bm25, chunker, hybrid, ingest, search


def test_package_is_importable() -> None:
    assert aico.__version__
    assert chunker.INGESTION_VERSION
    assert bm25.K1 > 0
    assert bm25.B > 0
    assert hybrid.RRF_K > 0
    assert model_gateway.ModelGateway


def test_ingest_cli_requires_arguments() -> None:
    try:
        ingest.main([])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("ingest CLI should reject missing arguments")


def test_search_cli_requires_query() -> None:
    try:
        search.main([])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("search CLI should reject missing --query")


def test_eval_cli_requires_arguments() -> None:
    try:
        day01.main([])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("eval CLI should reject missing arguments")


def test_day02_eval_cli_requires_arguments() -> None:
    try:
        day02.main([])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("day02 eval CLI should reject missing arguments")


def test_supplied_pack_is_in_data() -> None:
    documents = Path("data/documents")
    day01_queries = Path("data/evals/day01_queries.json")
    day02_queries = Path("data/evals/day02_queries.json")
    assert documents.is_dir()
    assert len(list(documents.glob("DOC-*.md"))) == 5
    assert day01_queries.is_file()
    assert day02_queries.is_file()
    assert hybrid.RRF_K == 60
