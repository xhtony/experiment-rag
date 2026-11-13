"""Phase 2 CLI. Explicit .env.phase2 selection; legacy scripts remain Phase 1."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-file", default=str(ROOT / ".env.phase2"))
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check configuration without spending API credits")
    doctor.add_argument("--services", action="store_true", help="Also check local PostgreSQL and Qdrant")
    sub.add_parser("init", help="Create isolated Phase 2 tables")
    ingest = sub.add_parser("ingest", help="Ingest only into empty isolated stores")
    ingest.add_argument("--corpus", type=Path, default=ROOT / "data/corpus")
    ingest.add_argument("--attest-corpus-reviewed", action="store_true")
    draft = sub.add_parser("prepare-questions", help="Convert legacy questions to UNREVIEWED development drafts")
    draft.add_argument("--source", type=Path, default=ROOT / "data/questions/questions.jsonl")
    draft.add_argument("--out", type=Path, required=True)
    schema = sub.add_parser("question-schema")
    schema.add_argument("--out", type=Path)
    validate = sub.add_parser("validate-questions", help="Check syntax/labels without databases or model calls")
    validate.add_argument("--file", type=Path, required=True)
    validate.add_argument("--split", choices=["development", "test"], required=True)
    load = sub.add_parser("load-questions", help="Register an immutable versioned question set")
    load.add_argument("--file", type=Path, required=True)
    load.add_argument("--name", required=True)
    load.add_argument("--split", choices=["development", "test"], required=True)
    load.add_argument("--development-file", type=Path, default=ROOT / "data/questions/questions.jsonl")
    load.add_argument("--attest-heldout", action="store_true")
    load.add_argument("--overlap-review", type=Path)
    freeze = sub.add_parser("freeze", help="Seal a reproducible run before calling any model")
    freeze.add_argument("--dataset", required=True)
    freeze.add_argument("--name", required=True)
    freeze.add_argument("--seed", type=int)
    freeze.add_argument("--conditions", nargs="+", choices=["C0", "C1", "C2", "C3", "C4"])
    run = sub.add_parser("run")
    run.add_argument("--run-id", type=int, required=True)
    run.add_argument("--retry-errors", action="store_true", help="Retry errored cells only if their frozen attempt budget is not exhausted")
    status = sub.add_parser("status")
    status.add_argument("--run-id", type=int, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--run-id", type=int, required=True)
    evaluate.add_argument("--allow-incomplete", action="store_true")
    evaluate.add_argument("--retry-errors", action="store_true",
                          help="Explicitly extend failed stage budgets; retain fixed claims and completed stages")
    manual = sub.add_parser("manual-export")
    manual.add_argument("--run-id", type=int, required=True)
    manual.add_argument("--out", type=Path, required=True)
    manual.add_argument("--sample-questions", type=int)
    manual.add_argument("--seed", type=int, default=42)
    imp = sub.add_parser("manual-import")
    imp.add_argument("--file", type=Path, required=True)
    imp.add_argument("--evaluator", required=True)
    imp.add_argument("--source", choices=["human", "ai_assisted"], default="human")
    imp.add_argument("--attest-human", action="store_true")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--run-id", type=int, required=True)
    analysis.add_argument("--out", type=Path, required=True)
    analysis.add_argument("--allow-incomplete", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from ragapp.phase2.environment import configure, require_api_key
    settings = configure(args.env_file)
    command = args.command
    if command == "doctor":
        from ragapp.phase2.provenance import runtime_manifest
        report = {"env_file": str(Path(args.env_file).resolve()), "generator": settings.llm_model,
                  "judge": settings.judge_model, "embedding": settings.embedding_model,
                  "api_key_available": bool(os.environ.get("OPENAI_API_KEY")),
                  "phase1_isolation": "passed", "runtime": runtime_manifest()}
        # Endpoint credentials and complete env contents are never printed.
        if args.services:
            from sqlalchemy import text
            from ragapp.storage.db import get_engine
            from ragapp.shared.vectorstore import get_qdrant_client
            with get_engine().connect() as c:
                c.execute(text("SELECT 1"))
            report["postgresql"] = "reachable"
            report["qdrant_collections"] = [x.name for x in get_qdrant_client().get_collections().collections]
        report["runtime"].pop("code", None)
        return report
    if command == "prepare-questions":
        from ragapp.phase2.datasets import prepare_development
        return {"draft_questions": prepare_development(args.source, args.out), "gold_reviewed": False}
    if command == "question-schema":
        from ragapp.phase2.contracts import QuestionRecord
        schema = QuestionRecord.model_json_schema()
        if args.out:
            if args.out.exists():
                raise ValueError("Schema output already exists")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
            return {"schema": str(args.out)}
        return schema
    if command == "validate-questions":
        from collections import Counter
        from ragapp.phase2.datasets import read_questions
        rows = read_questions(args.file)
        if args.split == "test" and any(not q.gold_reviewed for q in rows):
            raise ValueError("Held-out questions require audited gold annotations")
        return {"questions": len(rows), "qtype_counts": dict(Counter(q.qtype.value for q in rows)),
                "reviewed": sum(q.gold_reviewed for q in rows), "note": "Evidence IDs and overlap checked at load-questions"}
    if command == "init":
        from ragapp.storage.db import init_db
        from ragapp.phase2.models import init_phase2_db
        init_db()
        init_phase2_db()
        return {"phase2_tables": "created (existing rows retained)"}
    if command == "ingest":
        require_api_key()
        from ragapp.phase2.provenance import ingest_corpus
        return ingest_corpus(args.corpus, args.attest_corpus_reviewed)
    if command == "load-questions":
        from ragapp.phase2.provenance import verify_corpus
        from ragapp.phase2.datasets import import_dataset
        chunks, corpus_digest = verify_corpus()
        return {"dataset_id": import_dataset(args.file, args.name, args.split, chunks, corpus_digest,
                    args.development_file, args.attest_heldout, args.overlap_review)}
    if command == "freeze":
        from ragapp.phase2.runner import freeze_run
        return {"run_id": freeze_run(args.dataset, args.name, args.seed, args.conditions)}
    if command == "run":
        require_api_key()
        from ragapp.phase2.runner import run_experiment
        return run_experiment(args.run_id, retry_errors=args.retry_errors)
    if command == "status":
        from ragapp.phase2.runner import run_status
        return run_status(args.run_id)
    if command == "evaluate":
        require_api_key()
        from ragapp.phase2.workflow import evaluate_run
        return evaluate_run(args.run_id, args.allow_incomplete, retry_errors=args.retry_errors)
    if command == "manual-export":
        from ragapp.phase2.manual import export_manual
        from ragapp.phase2.runner import verify_run
        verify_run(args.run_id, check_live_corpus=False)
        return export_manual(args.run_id, args.out, args.sample_questions, args.seed)
    if command == "manual-import":
        from ragapp.phase2.manual import import_manual
        return {"imported": import_manual(args.file, args.evaluator, args.source, args.attest_human)}
    if command == "analyze":
        from ragapp.phase2.workflow import analyze_run
        return analyze_run(args.run_id, args.out, args.allow_incomplete)


if __name__ == "__main__":
    try:
        result = main()
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str))
        if isinstance(result, dict) and result.get("status") == "incomplete":
            sys.exit(2)
        if isinstance(result, dict) and result.get("evaluation_complete") is False:
            sys.exit(2)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Phase 2: {exc}", file=sys.stderr)
        sys.exit(2)
