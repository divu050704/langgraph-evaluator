from return_types import metric_type, test_type
from deepeval.metrics import TaskCompletionMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.integrations.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph
from deepeval.models import GPTModel, AnthropicModel, GeminiModel
from typing import List

def exec(app: CompiledStateGraph, inputs: List[str], output_parameter: str, input_parameter: str ,model: GPTModel | AnthropicModel | GeminiModel) -> test_type:
    relevancy_metrics: List[metric_type] = []
    completion_metrics: List[metric_type] = []
    for ginput in inputs:
        result = app.invoke(
            {input_parameter: ginput},
            config={"callbacks": [CallbackHandler()]},
        )

        test_case = LLMTestCase(
            input=ginput,
            actual_output=result[output_parameter],
        )

        relevancy_metric = AnswerRelevancyMetric(model=model)
        relevancy_metric.measure(test_case)
        return_relevancy: metric_type = {
            "reason": relevancy_metric.reason,
            "success": relevancy_metric.success,
            "score": relevancy_metric.score
        }

        completion_metric = TaskCompletionMetric(model=model)
        completion_metric.measure(test_case)
        return_completion: metric_type = {
            "reason": completion_metric.reason,
            "success": completion_metric.success,
            "score": completion_metric.score
        }
        relevancy_metrics.append(return_relevancy)
        completion_metrics.append(return_completion)
    return {"completion_metrics": completion_metrics, "relevancy_metrics": relevancy_metrics}