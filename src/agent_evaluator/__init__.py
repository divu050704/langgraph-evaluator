from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from typing import List
from response_types import *
from langgraph.graph.state import StateGraph


def run_test(app: StateGraph, topics: List[str]) -> run_test_response_type:
    relevancy_metrics: List[response_metric_type]
    completion_metrics: List[response_metric_type]
    for topic in topics:
        result = app.invoke(
            {"topic": topic},
            config={"callbacks": [CallbackHandler()]},
        )

        test_case = LLMTestCase(
            input=topic,
            actual_output=result["essay"],
        )

        print(f"\n=== Topic: {topic} ===")

        relevancy_metric_calculated = AnswerRelevancyMetric()
        relevancy_metric_calculated.measure(test_case)
        relevancy_metric: response_metric_type = {
            "score": relevancy_metric_calculated.score,
            "reason": relevancy_metric_calculated.reason,
            "success": relevancy_metric_calculated.success
        }

        completion_metric_calculated = TaskCompletionMetric()
        completion_metric_calculated.measure(test_case)
        completion_metric: response_metric_type = {
            "score": completion_metric_calculated.score,
            "reason": completion_metric_calculated.reason,
            "success": completion_metric_calculated.success
        }

        relevancy_metrics.append(relevancy_metric)
        completion_metrics.append(completion_metric)
    return {"completion_metric": completion_metrics, "relevancy_metric": relevancy_metric}