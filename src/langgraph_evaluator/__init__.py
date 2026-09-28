import sys
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

# Ensure src root is accessible for imports
_src_dir = str(Path(__file__).parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from return_types import metric_type, test_type, EvaluationResult, RegressionReport, MetricRegressionResult
from langgraph.graph.state import CompiledStateGraph
from eval_model import get_eval_model
from tracer import InMemoryTracer
import run_test
import create_tests
import storage
import stats
from storage import save_and_compare, resolve_pytest_node_id
from stats import (
    cliffs_delta,
    mann_whitney_u_test,
    fishers_exact_test,
    benjamini_hochberg,
    evaluate_regression_battery,
)


def test(
    app: CompiledStateGraph,
    inputs: Union[List[str], str],
    output_parameter: str,
    input_parameter: str,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    graph_name: Optional[str] = None,
    save_results: bool = True,
    alpha: float = 0.05,
    fail_on_regression: bool = False,
    results_dir: Union[Path, str] = "test_results",
) -> Union[EvaluationResult, test_type]:
    """
    Run evaluation on a LangGraph application, persist test metrics preserving
    raw score distributions, and conduct statistical regression testing against the previous run.

    Uses:
      - Pytest NodeID Convention for test target identification.
      - Mann-Whitney U test + Cliff's delta for continuous scores.
      - Fisher's exact test for binary pass/fail success rates.
      - Benjamini-Hochberg FDR correction across all tests.

    Args:
        app: Compiled LangGraph StateGraph instance.
        inputs: List of test strings or single seed input to generate test cases.
        output_parameter: Key name for actual output from graph result state.
        input_parameter: Key name for input parameter passed to graph.
        provider: Optional LLM provider (e.g., 'openai', 'anthropic', 'google').
        model_name: Optional LLM model identifier.
        graph_name: Optional explicit name override for the test/graph.
        save_results: If True, persists run and compares against previous baseline.
        alpha: Significance threshold for FDR-adjusted p-values (default: 0.05).
        fail_on_regression: If True, raises RuntimeError when statistical regression is detected.
    """
    model = get_eval_model(provider=provider, model_name=model_name)

    if isinstance(inputs, list):
        returned_metrics: test_type = run_test.exec(
            input_parameter=input_parameter,
            inputs=inputs,
            app=app,
            model=model,
            output_parameter=output_parameter,
        )
    else:
        tracer = InMemoryTracer()
        app.invoke(
            {input_parameter: inputs},
            config={"callbacks": [tracer]},
        )
        readable_traces = tracer.get_readable_runs()
        generated_inputs = create_tests.exec(model, readable_traces)
        returned_metrics = run_test.exec(
            input_parameter=input_parameter,
            inputs=generated_inputs,
            app=app,
            model=model,
            output_parameter=output_parameter,
        )

    if not save_results:
        return returned_metrics

    # Save run & conduct statistical regression testing using Pytest NodeID convention
    result: EvaluationResult = storage.save_and_compare(
        metrics=returned_metrics,
        app=app,
        graph_name=graph_name,
        alpha=alpha,
        results_dir=results_dir,
    )

    if fail_on_regression and result.get("regression_report", {}).get("has_regression"):
        reg_details = [
            f"- {k}: {v['reason']}"
            for k, v in result["regression_report"]["results"].items()
            if v.get("is_regression")
        ]
        msg = "Statistical performance regression detected in CI run:\n" + "\n".join(reg_details)
        raise RuntimeError(msg)

    return result


__all__ = [
    "test",
    "save_and_compare",
    "resolve_pytest_node_id",
    "cliffs_delta",
    "mann_whitney_u_test",
    "fishers_exact_test",
    "benjamini_hochberg",
    "evaluate_regression_battery",
    "metric_type",
    "test_type",
    "EvaluationResult",
    "RegressionReport",
    "MetricRegressionResult",
]