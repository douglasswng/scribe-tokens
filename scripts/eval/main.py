"""
Unified evaluation script for ScribeTokens models.

Supports:
- Single task evaluation: python -m scripts.eval.main --task HTR
- All tasks: python -m scripts.eval.main --all

Set the dataset in constants.py before running.
"""

import argparse

import scripts.eval.htg as htg
import scripts.eval.htr as htr
from ml_model.id import Task

TASK_EVALUATORS = {
    "HTR": htr.main,
    "HTG": htg.main,
}


def evaluate_task(task: str) -> None:
    """Evaluate a single task."""
    evaluator = TASK_EVALUATORS.get(task)
    if evaluator is None:
        print(f"No evaluator found for task: {task}")
        return

    print(f"\n{'=' * 60}")
    print(f"Evaluating {task}")
    print(f"{'=' * 60}")
    evaluator()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for ScribeTokens models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate HTR task
  python -m scripts.eval.main --task HTR

  # Evaluate all tasks
  python -m scripts.eval.main --all
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all tasks (HTR, HTG)",
    )
    mode_group.add_argument(
        "--task",
        type=str,
        choices=[task.value for task in Task if task.value in TASK_EVALUATORS],
        help="Specific task to evaluate",
    )

    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()

    if args.all:
        for task in TASK_EVALUATORS:
            evaluate_task(task)
    else:
        evaluate_task(args.task)

    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
