from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval import evaluate
from dotenv import load_dotenv

from common import (
    load_chat_log,
    load_goldens,
    match_chat_rows_to_goldens,
    get_expected_output,
)

load_dotenv()


def run():

    # Step 1: Load chat logs and Goldens
    chat_rows = load_chat_log()
    goldens = load_goldens()

    print("Questions in the log:", len(chat_rows))

    # Step 2: Match chat logs with Goldens
    matched = match_chat_rows_to_goldens(chat_rows, goldens)

    # Step 3: Create DeepEval test cases
    test_cases = []

    for row, golden in matched:

        expected_answer = get_expected_output(golden)

        if not expected_answer:
            print("Missing expected answer for:", row["input"])
            continue

        test_case = LLMTestCase(
            input=row["input"],
            actual_output=row["output"],
            expected_output=expected_answer,
        )

        test_cases.append(test_case)

    print("Test cases created:", len(test_cases))

    if not test_cases:
        raise ValueError("No matching Golden test cases found.")

    # Step 4: Configure Completeness metric
    metric = GEval(
        name="Answer Completeness",
        criteria=(
            "Evaluate whether the actual answer fully addresses the user's "
            "question and covers all essential information in the expected "
            "answer. Penalize missing important details, especially when "
            "the question has multiple parts. Do not penalize concise "
            "wording when all required information is present."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.8,
        model="gpt-4.1-mini",
    )

    # Step 5: Execute evaluation
    evaluate(test_cases=test_cases, metrics=[metric])

    print("Answer Completeness evaluation completed.")


if __name__ == "__main__":
    run()