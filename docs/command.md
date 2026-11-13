# 1. init qdrant and postgresql
python .\scripts\phase2.py init

# 2. ingest corpus into qdrant
python .\scripts\phase2.py ingest --attest-corpus-reviewed

# 3. freeze system before runing experiment
python .\scripts\phase2.py freeze --dataset test-1 --name phase2-test-1

# 4. run experiment
python .\scripts\phase2.py run --run-id 7  

## 4.1. check run status
python .\scripts\phase2.py status --run-id 7

{
  "run_id": 7,
  "name": "phase2-test-1",
  "status": "completed",
  "split": "test",
  "questions": 75,
  "conditions": [
    "C0",
    "C1",
    "C2",
    "C3",
    "C4"
  ],
  "expected": 375,
  "successful": 375,
  "technical_errors": 0,
  "not_attempted": 0,
  "automatically_scored": 0,
  "evaluation_records": 0,
  "evaluation_partial": 0,
  "evaluation_failed": 0,
  "evaluation_pending": 0,
  "evaluation_legacy_or_invalid": 0,
  "evaluation_not_started": 375,
  "evaluation_error_attempts": 0,
  "evaluation_error_types": [],
  "error_types": [
    "ValidationError"
  ]
}

# 5. evaluate experimental result
python .\scripts\phase2.py evaluate --run-id 7  
## 5.1. retry error result:
python .\scripts\phase2.py evaluate --run-id 7 --retry-errors

# 6. export csv for manual evaluation
python .\scripts\phase2.py manual-export --run-id 7 --out results/phase2/manual-test-1.csv

# 7. import manul evaluation score csv
scripts/phase2.py manual-import --file <csv> --evaluator <id> [--source human|ai_assisted] [--attest-human]

# 8. analyze
python .\scripts\phase2.py analyze --run-id 7 --out .\results\run-7