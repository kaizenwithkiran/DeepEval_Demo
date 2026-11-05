"""Run all five DeepEval metrics in sequence."""

from completeness import run as run_completeness
from faithfulness import run as run_faithfulness
from hallucination import run as run_hallucination
from relevancy import run as run_relevancy
from toxicity import run as run_toxicity


def main():
    evaluations = [
        ("Faithfulness (runtime retrieved context)", run_faithfulness),
        ("Answer Relevancy (no Goldens)", run_relevancy),
        ("Toxicity (no Goldens)", run_toxicity),
        ("Hallucination (Golden reference context)", run_hallucination),
        ("Answer Completeness (Golden expected answer)", run_completeness),
    ]

    failures = []
    for title, run_evaluation in evaluations:
        print("\n" + "=" * 72)
        print("Starting:", title)
        print("=" * 72)
        try:
            run_evaluation()
        except Exception as error:
            failures.append((title, error))
            print(f"FAILED: {title}: {type(error).__name__}: {error}")

    print("\n" + "=" * 72)
    print("EVALUATION RUN SUMMARY")
    print("=" * 72)
    if failures:
        for title, error in failures:
            print(f"FAILED - {title}: {type(error).__name__}: {error}")
        raise SystemExit(1)

    print("All five evaluations completed successfully.")


if __name__ == "__main__":
    main()
