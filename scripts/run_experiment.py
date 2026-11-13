"""Run the experiment: every question through every selected condition.

Usage:
    python scripts/run_experiment.py --name pilot-1                  # all 4 conditions
    python scripts/run_experiment.py --name h2 --conditions C1 C2    # subset
    python scripts/run_experiment.py --resume 3 --conditions C3      # resume run id 3
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from ragapp.schemas import Condition
from ragapp.experiment.runner import create_run, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, help="Name for a new run.")
    parser.add_argument("--resume", type=int, default=None, help="Existing run id to resume.")
    parser.add_argument(
        "--conditions", nargs="+", default=["C0", "C1", "C2", "C3"],
        choices=["C0", "C1", "C2", "C3"],
    )
    parser.add_argument("--questions", nargs="*", type=int, default=None,
                        help="Optional subset of question ids.")
    args = parser.parse_args()

    if args.resume is not None:
        run_id = args.resume
    else:
        if not args.name:
            parser.error("--name is required for a new run")
        run_id = create_run(args.name)
        print(f"Created run {run_id} ('{args.name}')")

    conditions = [Condition(c) for c in args.conditions]
    run_experiment(run_id, conditions, question_ids=args.questions)
    print(f"Done. Evaluate with: python scripts/evaluate.py --run-id {run_id}")


if __name__ == "__main__":
    main()
