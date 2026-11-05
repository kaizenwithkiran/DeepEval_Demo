import csv
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval import evaluate
from dotenv import load_dotenv

load_dotenv()


def run():

    # Step 1: Create empty lists
    inputs = []
    outputs = []
    contexts = []

    # Step 2: Read chatbot execution logs
    with open("chat_log.csv", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            inputs.append(row["input"])
            outputs.append(row["output"])
            contexts.append(row["context"])

    print("Questions in the log:", len(inputs))

    # Step 3: Create DeepEval test cases
    test_cases = []

    for i in range(len(inputs)):
        test_case = LLMTestCase(
            input=inputs[i],
            actual_output=outputs[i],
            retrieval_context=[contexts[i]],
        )

        test_cases.append(test_case)

    # Step 4: Configure Faithfulness metric
    metric = FaithfulnessMetric(
        threshold=0.8,
        model="gpt-4.1-mini",
        include_reason=True,
    )

    # Step 5: Execute evaluation
    evaluate(
        test_cases=test_cases,
        metrics=[metric]
    )

    # Step 6: Print completion message
    print("Faithfulness evaluation completed. Check the results in the output directory.")


if __name__ == "__main__":
    run()