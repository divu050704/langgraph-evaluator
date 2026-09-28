import os
import shutil
from pathlib import Path
import sys

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stats import (
    cliffs_delta,
    mann_whitney_u_test,
    fishers_exact_test,
    benjamini_hochberg,
    evaluate_regression_battery,
)
from storage import resolve_pytest_node_id, save_and_compare, sanitize_filename


def test_cliffs_delta():
    # 1. Clear drop
    before = [0.90, 0.92, 0.88, 0.95, 0.89, 0.94]
    after = [0.40, 0.45, 0.50, 0.42, 0.38, 0.48]
    delta, magnitude = cliffs_delta(after=after, before=before)
    assert delta == -1.0
    assert magnitude == "large"

    # 2. Equal distributions
    delta_eq, mag_eq = cliffs_delta(after=before, before=before)
    assert delta_eq == 0.0
    assert mag_eq == "negligible"


def test_mann_whitney_u_drop():
    before = [0.95, 0.92, 0.90, 0.89, 0.94, 0.91, 0.93, 0.96]
    after = [0.50, 0.55, 0.48, 0.52, 0.49, 0.53, 0.51, 0.47]
    p_val = mann_whitney_u_test(after=after, before=before)
    assert p_val < 0.01


def test_fishers_exact_drop():
    before = [True] * 15 + [False] * 1
    after = [True] * 3 + [False] * 13
    p_val, after_rate, before_rate = fishers_exact_test(after_successes=after, before_successes=before)
    assert p_val < 0.001
    assert after_rate < before_rate


def test_benjamini_hochberg_correction():
    raw_p = [0.001, 0.01, 0.04, 0.20, 0.80]
    adj_p = benjamini_hochberg(raw_p)
    assert len(adj_p) == len(raw_p)
    # Check monotonicity
    for i in range(len(adj_p) - 1):
        assert adj_p[i] <= adj_p[i + 1]
    # Adjusted p-values should be >= raw p-values
    for r, a in zip(raw_p, adj_p):
        assert a >= r


def test_evaluate_regression_battery_detection():
    before_metrics = {
        "faithfulness": [
            {"score": 0.92, "success": True},
            {"score": 0.95, "success": True},
            {"score": 0.90, "success": True},
            {"score": 0.88, "success": True},
            {"score": 0.94, "success": True},
            {"score": 0.91, "success": True},
        ],
        "schema_validation": [
            {"score": 1.0, "success": True},
            {"score": 1.0, "success": True},
            {"score": 1.0, "success": True},
            {"score": 1.0, "success": True},
            {"score": 1.0, "success": True},
        ],
    }

    # Candidate with a significant continuous regression in faithfulness
    after_metrics = {
        "faithfulness": [
            {"score": 0.45, "success": False},
            {"score": 0.50, "success": False},
            {"score": 0.42, "success": False},
            {"score": 0.48, "success": False},
            {"score": 0.40, "success": False},
            {"score": 0.44, "success": False},
        ],
        "schema_validation": [
            {"score": 1.0, "success": True},
            {"score": 1.0, "success": True},
            {"score": 1.0, "success": True},
            {"score": 1.0, "success": True},
            {"score": 1.0, "success": True},
        ],
    }

    report = evaluate_regression_battery(
        current_metrics=after_metrics,
        previous_metrics=before_metrics,
        alpha=0.05,
    )
    assert report["has_regression"] is True
    faith_result = report["results"]["faithfulness_continuous"]
    assert faith_result["is_regression"] is True
    assert faith_result["effect_size"] <= -0.147
    assert faith_result["adjusted_p_value"] < 0.05


def test_pytest_node_id_resolution(monkeypatch):
    # 1. Override takes precedence
    assert resolve_pytest_node_id(graph_name="custom_eval") == "custom_eval"

    # 2. Pytest env var convention
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/integration/test_agent.py::test_support_flow (call)")
    resolved = resolve_pytest_node_id()
    assert resolved == "tests_integration_test_agent__test_support_flow"


def test_save_and_compare_lifecycle(tmp_path):
    test_dir = tmp_path / "test_results"

    run_1_metrics = {
        "faithfulness": [
            {"score": 0.90, "success": True},
            {"score": 0.95, "success": True},
            {"score": 0.92, "success": True},
            {"score": 0.91, "success": True},
        ]
    }
    run_2_metrics = {
        "faithfulness": [
            {"score": 0.40, "success": False},
            {"score": 0.45, "success": False},
            {"score": 0.38, "success": False},
            {"score": 0.42, "success": False},
        ]
    }

    # Run 1: baseline
    res1 = save_and_compare(
        metrics=run_1_metrics,
        graph_name="test_graph",
        results_dir=test_dir,
    )
    assert res1["previous"] is None
    assert res1["regression_report"] is None

    # Run 2: regression candidate
    res2 = save_and_compare(
        metrics=run_2_metrics,
        graph_name="test_graph",
        results_dir=test_dir,
    )
    assert res2["previous"] is not None
    assert res2["regression_report"] is not None
    assert res2["regression_report"]["has_regression"] is True

    # Check files exist
    assert (test_dir / "test_graph_history.json").exists()
    assert (test_dir / "test_graph_performance.csv").exists()


if __name__ == "__main__":
    print("Running test_cliffs_delta...")
    test_cliffs_delta()
    print("Running test_mann_whitney_u_drop...")
    test_mann_whitney_u_drop()
    print("Running test_fishers_exact_drop...")
    test_fishers_exact_drop()
    print("Running test_benjamini_hochberg_correction...")
    test_benjamini_hochberg_correction()
    print("Running test_evaluate_regression_battery_detection...")
    test_evaluate_regression_battery_detection()
    
    # Test NodeID
    assert resolve_pytest_node_id(graph_name="custom_eval") == "custom_eval"
    os.environ["PYTEST_CURRENT_TEST"] = "tests/integration/test_agent.py::test_support_flow (call)"
    assert resolve_pytest_node_id() == "tests_integration_test_agent__test_support_flow"
    del os.environ["PYTEST_CURRENT_TEST"]
    
    # Test lifecycle
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_save_and_compare_lifecycle(Path(td))

    print("\nALL 7 TESTS PASSED SUCCESSFULLY!")

