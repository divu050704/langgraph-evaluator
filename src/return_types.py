from typing import TypedDict, List, Optional, Dict, Any

class metric_type(TypedDict):
    success: Optional[bool]
    score: Optional[float]
    reason: Optional[str]
    
class run_test_type(TypedDict):
    relevancy_metrics: metric_type
    completion_metric: metric_type

class test_type(TypedDict): 
    relevancy_metrics: List[metric_type]
    completion_metrics: List[metric_type]

class MetricRegressionResult(TypedDict):
    metric: str
    type: str  # "continuous" or "binary"
    test: str  # "Mann-Whitney U + Cliff's Delta" or "Fisher's Exact Test"
    raw_p_value: float
    adjusted_p_value: float
    effect_size: float
    effect_size_magnitude: str
    is_regression: bool
    reason: str

class RegressionReport(TypedDict):
    has_regression: bool
    alpha: float
    total_tests_conducted: int
    results: Dict[str, MetricRegressionResult]

class EvaluationResult(TypedDict, total=False):
    target_id: str
    current: Dict[str, Any]
    previous: Optional[Dict[str, Any]]
    diff: Optional[Dict[str, Any]]
    regression_report: Optional[RegressionReport]
    raw_current_metrics: test_type
    history_file: str
    csv_file: str