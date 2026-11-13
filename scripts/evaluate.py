"""Automated unified evaluation (LLM-as-judge) of a stored run.

Usage:
    python scripts/evaluate.py --run-id 1 [--conditions C1 C3]
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from ragapp.evaluation.evaluator import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--conditions", nargs="*", default=None,
                        choices=["C0", "C1", "C2", "C3"])
    args = parser.parse_args()

    n = evaluate_run(args.run_id, conditions=args.conditions)
    print(f"Scored {n} new answers for run {args.run_id}.")
    print(f"Analyze with: python scripts/analyze.py --run-id {args.run_id}")


if __name__ == "__main__":
    main()
