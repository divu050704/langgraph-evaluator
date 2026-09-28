from typing import TypedDict, List

class response_metric_type(TypedDict):
    score: float
    reason: str
    success: bool


class run_test_response_type(TypedDict):
    relevancy_metric: response_metric_type
    completion_metric: response_metric_type