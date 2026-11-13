"""Freeze actual code, document content, relational chunks and vector-index state."""

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from ragapp.config import get_settings
from ragapp.phase2.datasets import digest_json
from ragapp.phase2.environment import ROOT
from ragapp.phase2.models import CorpusState
from ragapp.schemas import RetrievedChunk
from ragapp.storage.db import session_scope
from ragapp.storage.models import Chunk, Document

PACKAGES = ("ragapp", "langchain", "langchain-core", "langchain-openai", "langchain-qdrant",
            "langgraph", "openai", "qdrant-client", "pydantic", "sqlalchemy", "psycopg2-binary",
            "rank-bm25", "numpy", "pandas", "scipy", "python-dotenv", "langchain-text-splitters")


def code_manifest():
    files = sorted((ROOT / "src" / "ragapp").rglob("*.py"))
    files += [ROOT / "scripts" / "phase2.py", ROOT / "requirements.txt", ROOT / "pyproject.toml"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    return {"sha256": digest_json(hashes), "files": hashes}


def runtime_manifest():
    settings = get_settings()
    from ragapp.shared.llm import chat_options
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    return {"settings": settings.snapshot(), "effective_generator": chat_options(settings.llm_model, settings.llm_temperature, settings.llm_reasoning_effort),
            "effective_judge": chat_options(settings.judge_model, 0.0, settings.judge_reasoning_effort),
            "code": code_manifest(), "packages": versions, "python": platform.python_version(), "git_commit": commit,
            "endpoint_fingerprint": digest_json([settings.database_url, settings.qdrant_url, settings.qdrant_collection,
                                                   os.environ.get("OPENAI_BASE_URL", "default")])}


def stored_chunks():
    with session_scope() as session:
        rows = session.query(Chunk, Document).join(Document, Chunk.document_id == Document.id).order_by(Chunk.id).all()
        chunks = [RetrievedChunk(chunk_id=c.id, document_title=d.title, section=c.section, text=c.text).model_dump() for c, d in rows]
    if not chunks:
        raise ValueError("No ingested chunks; run Phase 2 ingestion first")
    return chunks


def file_manifest(corpus: Path):
    files = {p.relative_to(corpus).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(corpus.rglob("*")) if p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf"}}
    if not files:
        raise ValueError("Corpus contains no supported documents")
    return files


def index_manifest(chunks: list[dict]):
    from ragapp.shared.vectorstore import get_qdrant_client
    settings = get_settings()
    client = get_qdrant_client()
    expected = {c["chunk_id"]: (c["text"], c["document_title"], c["section"]) for c in chunks}
    found, vectors, offset = {}, {}, None
    while True:
        points, offset = client.scroll(settings.qdrant_collection, limit=128, offset=offset,
                                       with_payload=True, with_vectors=True)
        for point in points:
            payload = point.payload or {}
            meta = payload.get("metadata", {})
            cid = meta.get("chunk_id")
            if not cid or cid in found:
                raise ValueError("Vector index has missing/duplicate chunk IDs")
            found[cid] = (payload.get("page_content"), meta.get("document_title"), meta.get("section", ""))
            if not point.vector:
                raise ValueError("Vector index contains an empty embedding")
            vectors[cid] = digest_json(point.vector)
        if offset is None:
            break
    if found != expected:
        raise ValueError("PostgreSQL and Qdrant chunk content differ; repair ingestion before running")
    return {"payload_sha256": digest_json(found), "vectors_sha256": digest_json(vectors), "count": len(found),
            "server_version": client.info().version}


def verify_corpus():
    with session_scope() as session:
        state = session.get(CorpusState, 1)
        if state is None:
            raise ValueError("No verified Phase 2 corpus manifest; use the Phase 2 ingest command")
        manifest = dict(state.manifest)
    settings = get_settings()
    chunks = stored_chunks()
    if manifest["chunks_sha256"] != digest_json(chunks):
        raise ValueError("Relational corpus changed after ingestion")
    if manifest["files"] != file_manifest(Path(manifest["corpus_dir"])):
        raise ValueError("Source corpus files changed; a new dataset/index version is required")
    for name in ("embedding_model", "chunk_size", "chunk_overlap"):
        if manifest[name] != getattr(settings, name):
            raise ValueError(f"Corpus configuration changed: {name}")
    if manifest["index"] != index_manifest(chunks):
        raise ValueError("Vector index changed after verification")
    return chunks, digest_json(manifest)


def ingest_corpus(corpus: Path, attest_corpus_reviewed: bool):
    if not attest_corpus_reviewed:
        raise ValueError("Confirm corpus permissions and personal-data review with --attest-corpus-reviewed")
    from ragapp.shared.corpus import ingest
    from ragapp.shared.vectorstore import get_qdrant_client
    from ragapp.storage.models import ExperimentRun
    settings = get_settings()
    files = file_manifest(corpus)
    with session_scope() as session:
        if session.get(CorpusState, 1) or session.query(Chunk).first() or session.query(ExperimentRun).first():
            raise ValueError("Ingestion only accepts an empty isolated database; existing data will not be overwritten")
    client = get_qdrant_client()
    if client.collection_exists(settings.qdrant_collection):
        if client.count(settings.qdrant_collection, exact=True).count:
            raise ValueError("Qdrant collection is not empty; existing vectors will not be overwritten")
    stats = ingest(corpus, recreate=False)
    chunks = stored_chunks()
    manifest = {"corpus_dir": str(corpus.resolve()), "files": files, "chunks_sha256": digest_json(chunks),
                "index": index_manifest(chunks), "embedding_model": settings.embedding_model,
                "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap,
                "permissions_and_personal_data_review_attested": True, "statistics": stats}
    if files != file_manifest(corpus):
        raise ValueError("Corpus changed during ingestion; manifest not sealed")
    with session_scope() as session:
        session.add(CorpusState(id=1, manifest=manifest))
    return stats
