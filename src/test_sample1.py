from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="How do I add new notes?",
        actual_output="What is this project built with?"
    )
    assert_test(test_case, [answer_relevancy_metric])
