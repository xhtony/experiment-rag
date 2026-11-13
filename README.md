# RAG Strategy Evaluation System for University Course & Policy QA

## Phase 2

The isolated C0-C4 workflow is now available through `scripts/phase2.py`, which
explicitly reads `.env.phase2`. See [Phase 2 protocol and runbook](docs/phase2-workflow.md)
for development/held-out question sets, configuration freezing, execution,
automatic evaluation, blinded human scoring, and paired statistical analysis.
Run `python scripts/phase2.py doctor` for a non-billable configuration check.
The documentation below describes the preserved Phase 1 workflow.

Experimental system for the thesis topic:

> **Evaluating Retrieval-Augmented Generation Strategies for University Course and
> Policy Question Answering: Faithfulness, Citation Accuracy, Hallucination, and
> Refusal Behaviour**

The design follows *shared infrastructure + multiple RAG strategy variants +
unified evaluation*: four independent LangGraph pipelines run over the same
corpus, embeddings, LLM, and prompt scaffolding; their outputs land in one
PostgreSQL schema and are scored by one evaluator.

## Architecture

```
                      ┌─────────────────────────────────────────────┐
                      │            Shared infrastructure            │
   data/corpus ──────►│ ingestion → chunks → Qdrant (dense vectors) │
                      │                    → PostgreSQL (BM25 src)  │
                      │ OpenAI LLM (one model, temp=0) + embeddings │
                      └──────────────┬──────────────────────────────┘
                                     │
        ┌──────────────┬─────────────┴───────────┬──────────────────┐
        ▼              ▼                         ▼                  ▼
  C0 no-RAG      C1 vector RAG          C2 hybrid + rerank   C3 corrective/refusal
  generate       retrieve→generate      dense+BM25→RRF       retrieve→grade evidence
                                        →LLM rerank→generate →(rewrite→retry)→generate
                                                             →verify→answer/refuse
        └──────────────┴──────────────┬──────────┴──────────────────┘
                                      ▼
                      PostgreSQL: answers (full trace per question×condition)
                                      ▼
            Unified evaluation (LLM-as-judge + gold labels + manual rubric)
            faithfulness · citation P/R/F1 · hallucination · refusal behaviour
                                      ▼
            Statistical analysis: Friedman, Wilcoxon+Holm, Cliff's δ,
            Kendall's W, Spearman/Pearson/κ/ICC (auto vs manual, H5)
```

### Experimental conditions

| Condition | Graph | Manipulation vs C1 |
|---|---|---|
| C0 | `generate` | no retrieval at all |
| C1 | `retrieve → generate` | — (baseline) |
| C2 | `hybrid_retrieve → rerank → generate` | retrieval: dense+BM25 RRF fusion + LLM reranker |
| C3 | `retrieve → grade_evidence → [rewrite→retry] → generate → verify / refuse` | evidence sufficiency check, refusal-aware prompt, post-generation verification |

Controlled variables: same LLM + temperature, same corpus/chunking, same final
`top_k` context size, same structured-output answer format (sentence-level
citations), same base persona prompt. The full `Settings` snapshot is frozen
into each `experiment_runs` row.

## Setup

```bash
# 1. Infrastructure (Qdrant + PostgreSQL)
docker compose up -d

# 2. Python environment (3.10+)
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
pip install -e .

# 3. Configuration
copy .env.example .env                              # then set OPENAI_API_KEY
```

## Workflow

```bash
# 1. Create tables
python scripts/init_db.py

# 2. Ingest the controlled corpus (PostgreSQL + Qdrant)
python scripts/ingest.py --corpus data/corpus --show-chunks
#    --show-chunks prints chunk ids → use them to label gold_chunk_ids

# 3. Load the gold-standard question set
python scripts/load_questions.py --file data/questions/questions.jsonl --replace

# 4. Run the experiment (repeated measures: every question × every condition)
python scripts/run_experiment.py --name pilot-1
#    resumable: python scripts/run_experiment.py --resume <run_id>

# 5. Automated unified evaluation (LLM-as-judge)
python scripts/evaluate.py --run-id 1

# 6. Manual rubric evaluation (blinded, randomized export)
python scripts/manual_eval.py export --run-id 1 --out manual_run1.csv --sample 60
python scripts/manual_eval.py import --file manual_run1.csv --evaluator alice
python scripts/manual_eval.py import --file manual_run1_bob.csv --evaluator bob

# 7. Statistical analysis → results/*.csv + hypothesis-test summary
python scripts/analyze.py --run-id 1
```

## How metrics map to the methodology

| Dependent variable | Implementation |
|---|---|
| Faithfulness | judge decomposes the answer into atomic claims, verifies each against the retrieved context (`supported`/`partial`/`unsupported`/`contradicted`); score = mean weight. For C0 (no retrieval) claims are verified against the gold evidence chunks. |
| Citation accuracy | precision/recall/F1 of cited chunk ids vs gold supporting chunks, plus sentence-level *citation support* (does each cited chunk actually support its sentence, judged by LLM). Fabricated chunk ids count against precision. |
| Hallucination | answer flagged if ≥1 unsupported or contradicted claim; rate = flagged answers / non-refused answers. |
| Refusal behaviour | each answer classified as `correct_answer` / `correct_refusal` / `false_refusal` / `false_answer` against the gold `expect_refusal` label; accuracy, precision, recall, F1, false-refusal and false-answer rates per condition. |
| Retrieval diagnostics | gold-chunk hit and recall per answer, supporting error analysis (retrieval vs ranking vs generation failure). |

### Hypothesis → analysis mapping

| Hypothesis | Where |
|---|---|
| H1 RAG > no-RAG faithfulness | `analyze.py`: faithfulness pairwise C0 vs C1/C2/C3 |
| H2 hybrid+rerank improves citations | citation metrics pairwise C1 vs C2 |
| H3 corrective RAG reduces hallucination | hallucination rate pairwise C1 vs C3 |
| H4 refusal accuracy on unanswerable items | refusal table on `expect_refusal` subset |
| H5 auto vs manual agreement | Spearman/Pearson on faithfulness, Cohen's κ on hallucination, ICC across evaluators |

Statistics: Shapiro-Wilk (reported), Friedman omnibus, pairwise Wilcoxon
signed-rank with Holm-Bonferroni correction, Cliff's delta, Kendall's W —
matching the repeated-measures, non-parametric analysis plan.

## Project layout

```
src/ragapp/
  config.py                 frozen experimental settings (.env-driven)
  schemas.py                AnswerPayload contract between pipelines & evaluation
  shared/                   corpus ingestion, Qdrant, BM25, LLM factories, prompts
  pipelines/                c0_no_rag · c1_vector_rag · c2_hybrid_rerank ·
                            c3_corrective_refusal · registry (LangGraph graphs)
  experiment/runner.py      repeated-measures runner (idempotent, resumable)
  evaluation/               LLM-as-judge primitives + unified evaluator
  analysis/stats.py         Friedman, Wilcoxon+Holm, effect sizes, κ/ICC
  storage/                  SQLAlchemy schema (PostgreSQL)
scripts/                    CLI entry points for every workflow step
data/corpus/                controlled corpus (markdown + YAML front matter, or PDF)
data/questions/             gold-standard question dataset (JSONL)
```

## Notes for the thesis write-up

- **Reproducibility**: every run stores a config snapshot; chunk ids are
  deterministic (`<doc-slug>::<seq>`); temperature 0; the manual-eval export is
  blinded (condition hidden) with a fixed random order across evaluators.
- **Gold labelling**: after any change to corpus or chunking parameters, re-run
  `ingest.py --show-chunks` and re-verify `gold_chunk_ids`.
- **Study data**: `data/corpus` contains a real study corpus built from public
  Auckland Institute of Studies (ais.ac.nz) and NZQA pages plus two internal
  COMP901 course documents (15 documents, 101 chunks), and
  `data/questions/questions.jsonl` contains 103 gold-labelled questions
  across the 5 types - see
  [docs/ais-corpus-and-questions.md](docs/ais-corpus-and-questions.md).
  The original fictional smoke-test set lives in `data/corpus-sample/` and
  `data/questions/sample-questions.jsonl`. Use
  `python scripts/preview_chunks.py` to verify chunk ids when labelling gold
  chunks (no database required).
- **Costs**: evaluation makes several judge calls per answer (claim
  decomposition, claim verification, per-citation support). Pilot with a
  question subset first (`run_experiment.py --questions 1 2 3`).
- **Ethics**: use only public university documents; no student data is stored.
