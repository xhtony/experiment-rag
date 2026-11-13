# experiment-rag

Experimental software, data and results for the master's thesis *Answer, Clarify or Refuse? Evaluating Retrieval and Answer-Control Strategies for Reliable Question Answering over University Course and Policy Documents* (COMP901 Information Technology Research Thesis, Auckland Institute of Studies, 2026).

The Python package `ragapp` compares retrieval-augmented generation (RAG) conditions on a fixed corpus of course and policy documents:

| Condition | Retrieval | Answer control | Phase |
|---|---|---|---|
| C0 | none | none | 1, 2 |
| C1 | dense (Qdrant) | none | 1, 2 |
| C2 | dense + BM25, RRF fusion, LLM reranking | none | 1, 2 |
| C3 | dense | corrective control: evidence grading, query rewrite, answer verification, refusal | 1, 2 |
| C4 | hybrid, as C2 | corrective control, as C3 | 2 |

The study ran in two phases with separate workflows:

- **Phase 1** (development study): 200 questions under C0–C3, run with the scripts in `scripts/` other than `phase2.py`. It uses `.env`, `docker-compose.yml` and its own database tables.
- **Phase 2** (held-out evaluation): 75 questions under C0–C4, run with `scripts/phase2.py`. It uses `.env.phase2` and `docker-compose.phase2.yml`. It refuses to start if its database or vector index points to the Phase 1 stores.

## Repository layout

```
src/ragapp/
  config.py                 settings read from .env or .env.phase2 (prefix RAGAPP_)
  schemas.py                conditions C0-C4 and the answer contract
  shared/                   corpus parsing and chunking, Qdrant store, BM25, model factories, Phase 1 prompts
  pipelines/                Phase 1 LangGraph pipelines C0-C3 and the condition registry
  experiment/runner.py      Phase 1 runner (every question under every condition, resumable)
  evaluation/               Phase 1 LLM-as-judge primitives and unified evaluator
  analysis/stats.py         Phase 1 statistics
  storage/                  SQLAlchemy models and database sessions (PostgreSQL)
  phase2/                   Phase 2: configuration isolation, datasets, provenance and freezing,
                            pipelines C0-C4, runner, staged evaluation, manual rating packets, analysis
scripts/
  init_db.py  ingest.py  load_questions.py  run_experiment.py
  evaluate.py  manual_eval.py  analyze.py  analyze.sql      Phase 1 workflow
  preview_chunks.py         chunk-ID preview without databases
  phase2.py                 Phase 2 command-line interface
  score_second_rater.py     scoring of the second-rater workbooks
tests/                      tests of the Phase 2 workflow (in-memory SQLite)
data/corpus/                corpus documents (Markdown with YAML front matter)
data/questions/phase1/      Phase 1 questions (questions.jsonl, 200 items)
data/questions/phase2/      Phase 2 held-out questions (questions.jsonl, 75 items) and their JSON schema
data/research_questions/    research questions and hypotheses of each phase
results/phase1/             Phase 1 result files of run 2
results/phase2/             Phase 2 result files of run 7 (the held-out run)
question set reliability/   second-rater workbooks, researcher keys and scored reports
docs/command.md             command log of the Phase 2 held-out run
docker-compose.yml          PostgreSQL (port 5432) and Qdrant (port 6333) for Phase 1
docker-compose.phase2.yml   PostgreSQL (port 5433) and Qdrant (port 6335) for Phase 2
.env.example                Phase 1 configuration template
.env.phase2.example         Phase 2 configuration template
```

## Corpus availability

The study corpus has 23 documents. Two of them are internal COMP901 course documents that are not redistributed: `comp901-thesis-assignment.md` and `comp901-thesis-marking-guide.md`. `data/corpus` therefore contains 21 documents. Without the two files:

- **Phase 1:** questions 41–54 and 131 cite chunks of these documents as gold evidence. Phase 1 still runs, but these questions lose their gold evidence.
- **Phase 2:** `load-questions` stops with `Unknown reference chunk IDs`, because held-out questions 214, 215, 227, 228, 229, 261 and 262 cite chunks of these documents.

Readers who have access to the two documents can place them in `data/corpus` under these file names before ingestion. This reproduces the 23-document corpus of the study.

## Requirements

- Python 3.10 or later (the held-out run used Python 3.13.13)
- Docker with Docker Compose
- An OpenAI API key. The study used `gpt-5.4-mini` as generator, `gpt-5.4` as judge and `text-embedding-3-small` for embeddings.

## Installation

Run all commands from the repository root.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` makes `ragapp` importable for the Phase 1 scripts and the tests. `phase2.py` adds `src` to the import path itself.

The commands below are single lines with forward slashes, so they run in PowerShell and in bash. Replace every placeholder in angle brackets, such as `<run_id>`, before running a command.

## Phase 1: development study

### 1. Services and configuration

```bash
docker compose up -d
```

Copy `.env.example` to `.env`. In PowerShell use `Copy-Item .env.example .env`; in bash use `cp .env.example .env`. Then set `OPENAI_API_KEY`.

The template's default models (`gpt-4o-mini`, `gpt-4o`) are not those of the study. To reproduce the study, set:

```
RAGAPP_LLM_MODEL=gpt-5.4-mini
RAGAPP_LLM_TEMPERATURE=0.0
RAGAPP_JUDGE_MODEL=gpt-5.4
RAGAPP_EMBEDDING_MODEL=text-embedding-3-small
```

The retrieval settings in the template are those of the study: chunk size 800, overlap 120, `top_k` 5 and `candidate_k` 20.

### 2. Commands

```bash
# create the Phase 1 tables
python scripts/init_db.py

# optional: list the chunk IDs that ingestion will produce, without databases
python scripts/preview_chunks.py --corpus data/corpus

# ingest the corpus into PostgreSQL and Qdrant (rebuilds the index; --keep appends instead)
python scripts/ingest.py --corpus data/corpus

# load the 200 development questions (the script's default path no longer exists, so give --file)
python scripts/load_questions.py --file data/questions/phase1/questions.jsonl --replace

# generate answers for every question under C0-C3; prints the new run id
python scripts/run_experiment.py --name phase1

# resume an interrupted run
python scripts/run_experiment.py --resume <run_id>

# automated evaluation (LLM-as-judge)
python scripts/evaluate.py --run-id <run_id>

# manual rubric rating: export a blinded, shuffled sheet, fill it in, import it
python scripts/manual_eval.py export --run-id <run_id> --out manual_run<run_id>.csv
python scripts/manual_eval.py import --file manual_run<run_id>.csv --evaluator <rater_id>

# statistics: run<run_id>_scores.csv, _descriptives.csv, _refusal.csv and _tests.json
python scripts/analyze.py --run-id <run_id> --out results/phase1-rerun
```

Useful options:

- `run_experiment.py --conditions C1 C2` runs a subset of conditions.
- `run_experiment.py --questions 1 2 3` runs a subset of question IDs.
- `evaluate.py --conditions C3` scores one condition only.
- `manual_eval.py export --sample 60` exports a random sample of answers.

In the manual sheet:

- `faithfulness_rubric` takes 2 (fully supported), 1 (partial) or 0 (unsupported).
- `citation_correct`, `hallucinated` and `refusal_correct` take 1 or 0.
- `error_category` and `notes` are optional.

The study's Phase 1 run is run 2. Its outputs are in `results/phase1` (`descriptives.csv`, `refusal.csv`, `scores.csv` and `tests.json`).

`scripts/analyze.sql` reproduces in SQL the parts of `analyze.py` that do not need SciPy: the score table, descriptives, refusal metrics, paired means and Cliff's delta. It analyses run 2; to analyse another run, edit the `run_id` in its first view. Then run it in bash:

```bash
docker compose exec -T postgres psql -U ragapp -d ragapp < scripts/analyze.sql
```

In PowerShell:

```powershell
Get-Content scripts/analyze.sql | docker compose exec -T postgres psql -U ragapp -d ragapp
```

## Phase 2: held-out evaluation

### 1. Services and configuration

```bash
docker compose -f docker-compose.phase2.yml up -d
```

Copy `.env.phase2.example` to `.env.phase2` and set `OPENAI_API_KEY`. The other values in the template are the frozen settings of the held-out run:

- generator `gpt-5.4-mini` and judge `gpt-5.4`, both at temperature 0 with reasoning effort `none`;
- chunk size 800 and overlap 120;
- `top_k` 5, `candidate_k` 20 and RRF k 60;
- a context limit of 12,000 characters and one corrective rewrite;
- seed 42 and at most two attempts per cell.

`phase2.py` reads `.env.phase2` by default. Another file can be given with `--env-file <file>` before the subcommand, for example:

```bash
python scripts/phase2.py --env-file <file> doctor
```

It never reads the Phase 1 `.env`.

Check the configuration without API costs:

```bash
python scripts/phase2.py doctor
python scripts/phase2.py doctor --services
```

### 2. Held-out run

These are the commands of the study's held-out run (dataset `test-1`, run `phase2-test-1`, which received run id 7):

```bash
python scripts/phase2.py init
python scripts/phase2.py ingest --attest-corpus-reviewed
python scripts/phase2.py validate-questions --file data/questions/phase2/questions.jsonl --split test
python scripts/phase2.py load-questions --file data/questions/phase2/questions.jsonl --name test-1 --split test --development-file data/questions/phase1/questions.jsonl --attest-heldout
python scripts/phase2.py freeze --dataset test-1 --name phase2-test-1
python scripts/phase2.py run --run-id <run_id>
python scripts/phase2.py status --run-id <run_id>
python scripts/phase2.py evaluate --run-id <run_id>
python scripts/phase2.py manual-export --run-id <run_id> --out manual/phase2-test-1.csv
python scripts/phase2.py manual-import --file manual/phase2-test-1.csv --evaluator <rater_id> --source human --attest-human
python scripts/phase2.py analyze --run-id <run_id> --out results/phase2-rerun
```

What each step does and checks:

| Command | Notes |
|---|---|
| `init` | Creates the Phase 2 tables; existing rows are kept. |
| `ingest` | Embeds and indexes the corpus and seals a corpus manifest. It needs the API key and accepts only an empty Phase 2 database and collection. `--attest-corpus-reviewed` confirms that corpus permissions and personal data were reviewed. |
| `validate-questions` | Checks syntax and labels without databases or model calls. Held-out items must have `gold_reviewed: true`. |
| `load-questions` | Registers an immutable, versioned dataset after verifying the corpus and every referenced chunk ID. A held-out set requires `--development-file` and `--attest-heldout` and is screened for overlap with earlier datasets. If near duplicates are flagged, a review file must be supplied with `--overlap-review <review.json>` containing `reviewer`, `rationale` and the `flags_digest` printed in the error message. Dataset names cannot be reused. |
| `freeze` | Seals the settings, code checksums, corpus, dataset, execution schedule and seed before any model call, and prints the run id. A held-out run must include all five conditions, a held-out dataset can have only one sealed run, and the judge must differ from the generator. |
| `run` | Executes the frozen schedule and can be resumed by issuing the same command again. `--retry-errors` retries failed cells within the frozen attempt budget. |
| `status` | Reports completed, failed, pending and evaluated cells. The script exits with code 2 while the run is incomplete. |
| `evaluate` | Runs the staged automated judge. `--retry-errors` retries failed evaluation stages and `--allow-incomplete` evaluates a run that is not complete. The script exits with code 2 while the evaluation is incomplete. |
| `manual-export` | Writes a blinded rating packet shuffled with `--seed` (default 42). `--sample-questions <n>` draws a stratified sample of questions with all their conditions. An existing file is never overwritten. |
| `manual-import` | Imports one completed packet. The packet columns must stay unchanged. Ratings by an independent person use `--source human --attest-human`; AI-assisted ratings use `--source ai_assisted`. |
| `analyze` | Writes the prespecified paired contrasts, descriptive tables, agreement statistics and `audit.json` to a new or empty folder (the same file set as `results/phase2`). `--allow-incomplete` analyses an incomplete run. |

In the study, all 375 responses of run 7 were rated manually. `results/phase2` contains the output of `analyze` for that run.

### 3. Development datasets (optional)

`prepare-questions` converts Phase 1 questions into unreviewed drafts. After review, the drafts can be loaded as a development set. Unlike a held-out set, a development set may be frozen with a subset of conditions:

```bash
python scripts/phase2.py prepare-questions --source data/questions/phase1/questions.jsonl --out <draft.jsonl>
python scripts/phase2.py validate-questions --file <reviewed.jsonl> --split development
python scripts/phase2.py load-questions --file <reviewed.jsonl> --name <dev_name> --split development
python scripts/phase2.py freeze --dataset <dev_name> --name <run_name> --conditions C3 C4
```

`python scripts/phase2.py question-schema` prints the JSON schema of a question record. Add `--out <file>` to write it to a new file.

### 4. Starting Phase 2 again on the same machine

`ingest` refuses to write into a Phase 2 database or collection that already holds data. To start again, back up the database first. Removing the volumes permanently deletes the Phase 2 database and vector index, including every sealed run.

```bash
docker compose -f docker-compose.phase2.yml exec postgres pg_dump -U ragapp -d ragapp -f /tmp/phase2_backup.sql
docker compose -f docker-compose.phase2.yml cp postgres:/tmp/phase2_backup.sql ./phase2_backup.sql
docker compose -f docker-compose.phase2.yml down -v
docker compose -f docker-compose.phase2.yml up -d
```

## Second-rater check of the question sets

The second rater worked in two rounds:

- **Round 1:** the 75 held-out questions.
- **Round 2:** a stratified sample of 50 development questions.

Replace `<rater>` with the suffix of the workbook names in `question set reliability`. The script needs `openpyxl` (`pip install openpyxl`).

```bash
python scripts/score_second_rater.py --key "question set reliability/RESEARCHER_KEY_Round1_Phase2_do_not_send.xlsx" --part-a "question set reliability/Round1_Phase2_PartA_blind_<rater>.xlsx" --part-b "question set reliability/Round1_Phase2_PartB_goldcheck_<rater>.xlsx" --out round1_phase2_report_rerun.xlsx
python scripts/score_second_rater.py --key "question set reliability/RESEARCHER_KEY_Round2_Phase1_do_not_send.xlsx" --part-a "question set reliability/Round2_Phase1_PartA_blind_<rater>.xlsx" --part-b "question set reliability/Round2_Phase1_PartB_goldcheck_<rater>.xlsx" --out round2_phase1_report_rerun.xlsx
```

The reports of the study are `round1_phase2_report.xlsx` and `round2_phase1_report.xlsx` in the same folder.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Notes

- `.env` and `.env.phase2` hold API keys and are excluded by `.gitignore`. Do not commit them.
- The experiment databases and vector indices are not part of this repository. Re-running a phase calls the models again. Because hosted models are not fully deterministic at temperature 0, new responses can differ from the archived ones.
