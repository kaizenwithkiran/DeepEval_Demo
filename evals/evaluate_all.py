from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    HallucinationMetric,
    ToxicityMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams
from common import load_chat_data, create_test_cases, print_usage_summary

inputs, outputs, contexts, token_costs, latencies = load_chat_data()
test_cases = create_test_cases(inputs, outputs, contexts)

metrics = [
    FaithfulnessMetric(
        threshold=0.8, model="gpt-4.1-mini", include_reason=True
    ),
    AnswerRelevancyMetric(
        threshold=0.7, model="gpt-4.1-mini", include_reason=True
    ),
    HallucinationMetric(
        threshold=0.5, model="gpt-4.1-mini", include_reason=True
    ),
    ToxicityMetric(
        threshold=0.5, model="gpt-4.1-mini", include_reason=True
    ),
    GEval(
        name="Answer Completeness",
        criteria=(
            "Determine whether the actual output fully answers the user's input. "
            "Check every part of the request, essential details, and missing sub-questions. "
            "Do not reward irrelevant length. If context is insufficient, the answer "
            "should clearly state the limitation instead of inventing details."
        ),
        evaluation_steps=[
            "Identify all requests and sub-questions in the input.",
            "Check whether the answer addresses each request.",
            "Identify essential missing information.",
            "Score from 0 to 1, where 1 means fully complete."
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
        model="gpt-4.1-mini"
    )
]

evaluate(test_cases=test_cases, metrics=metrics)
print_usage_summary(token_costs, latencies)
