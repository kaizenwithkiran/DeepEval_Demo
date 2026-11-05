from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

from common import context_to_list, load_chat_log, load_goldens, match_chat_rows_to_goldens


def run():
    # Step 1: Read actual chats and the Golden dataset.
    rows = load_chat_log(required_columns=("input", "output"))
    goldens = load_goldens()
    matched_rows = match_chat_rows_to_goldens(rows, goldens)

    # Step 2: Create test cases using the Golden context as ground truth.
    test_cases = []
    for row, golden in matched_rows:
        golden_context = context_to_list(golden.get("context"))
        if not golden_context:
            print("Skipping Golden without context:", row.get("input"))
            continue

        test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["output"],
            context=golden_context,
        )
        test_cases.append(test_case)

    print("Test cases with Golden context:", len(test_cases))
    if not test_cases:
        raise ValueError(
            "No chat rows matched a Golden with a non-empty context. "
            "Check input/id matching and the context field in goldens.json."
        )

    # Step 3: Create the judge metric.
    # HallucinationMetric compares the answer with trusted Golden context.
    metric = HallucinationMetric(
        threshold=0.8,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    # Step 4: Evaluate.
    evaluate(test_cases=test_cases, metrics=[metric])
    print("Hallucination evaluation completed.")


if __name__ == "__main__":
    run()
