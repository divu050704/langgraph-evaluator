from __future__ import annotations

import csv
import inspect
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import stats

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


RESULTS_DIR = Path("test_results")


def sanitize_filename(name: str) -> str:
    """Make a string safe for use as a cross-platform filename, preserving double underscores as delimiters."""
    # Replace path separators
    clean = name.replace("/", "_").replace("\\", "_")
    # Replace non-alphanumeric, dash, or underscore characters with _
    clean = re.sub(r"[^\w\-]", "_", clean)
    # Collapse 3 or more consecutive underscores into double underscore
    clean = re.sub(r"_{3,}", "__", clean).strip("_")
    return clean or "test_run"


def resolve_pytest_node_id(
    app: Optional[CompiledStateGraph] = None, graph_name: Optional[str] = None
) -> str:
    """
    Resolve a stable test/graph identifier following the Pytest NodeID Convention:
    1. Explicit graph_name parameter (Highest priority).
    2. app.name if specified and not the LangGraph default placeholder.
    3. PYTEST_CURRENT_TEST environment variable if running under pytest.
    4. Call stack inspection: caller test file (relative to cwd) + caller function.
    5. Fallback: 'graph'.
    """
    # 1. Explicit user override
    if graph_name:
        return sanitize_filename(graph_name)

    # 2. Named LangGraph application
    if app is not None:
        name = getattr(app, "name", None)
        if name and name != "LangGraph":
            return sanitize_filename(name)

    # 3. Running under Pytest (PYTEST_CURRENT_TEST = "tests/test_foo.py::test_bar (call)")
    pytest_current = os.environ.get("PYTEST_CURRENT_TEST")
    if pytest_current:
        # Strip trailing phase like ' (call)', ' (setup)', etc.
        test_id = pytest_current.split(" ")[0]
        # Remove .py before turning colons into double underscores
        test_id = re.sub(r"\.py\b", "", test_id)
        test_id = test_id.replace("::", "__")
        return sanitize_filename(test_id)



    # 4. Caller stack inspection
    cwd = Path.cwd().resolve()
    for frame_info in inspect.stack():
        frame_file = Path(frame_info.filename).resolve()
        # Skip internal library frames
        if "langgraph_evaluator" in str(frame_file) or frame_file.name in (
            "storage.py",
            "stats.py",
            "run_test.py",
            "__init__.py",
        ):
            continue

        try:
            rel_file = frame_file.relative_to(cwd)
        except ValueError:
            rel_file = Path(frame_file.stem)

        # Remove .py extension and join with function name
        file_slug = str(rel_file.with_suffix("")).replace(os.sep, "_")
        func_name = frame_info.function

        if func_name and func_name != "<module>":
            return sanitize_filename(f"{file_slug}__{func_name}")
        else:
            return sanitize_filename(file_slug)

    # 5. Safe fallback
    return "graph"


def _flatten_metrics_summary(metrics: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Generates scalar averages and success rates for human-readable CSV logging."""
    row: Dict[str, Any] = {"timestamp": datetime.now().isoformat(timespec="seconds")}
    for key, entries in metrics.items():
        if not entries:
            continue
        scores = [float(e["score"]) for e in entries if e.get("score") is not None]
        successes = [bool(e["success"]) for e in entries if e.get("success") is not None]
        row[f"{key}_avg_score"] = round(sum(scores) / len(scores), 4) if scores else None
        row[f"{key}_success_rate"] = round(sum(successes) / len(successes), 4) if successes else None
    return row


def save_and_compare(
    metrics: Dict[str, List[Dict[str, Any]]],
    app: Optional[CompiledStateGraph] = None,
    graph_name: Optional[str] = None,
    alpha: float = 0.05,
    results_dir: Path | str = RESULTS_DIR,
) -> Dict[str, Any]:
    """
    Persists full test distributions and performs regression analysis
    against the previous run using:
    - Mann-Whitney U test + Cliff's delta for continuous scores.
    - Fisher's exact test for binary pass/fail success rates.
    - Benjamini-Hochberg FDR correction across all tests.
    - Pytest NodeID Convention for test target naming.
    """
    target_dir = Path(results_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    target_id = resolve_pytest_node_id(app=app, graph_name=graph_name)
    history_json_path = target_dir / f"{target_id}_history.json"
    performance_csv_path = target_dir / f"{target_id}_performance.csv"

    # Load existing historical runs
    history: List[Dict[str, Any]] = []
    if history_json_path.exists():
        try:
            with open(history_json_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    previous_run = history[-1] if history else None
    previous_metrics = previous_run.get("metrics") if previous_run else None

    # Run statistical regression battery if previous run exists
    regression_report = None
    if previous_metrics:
        regression_report = stats.evaluate_regression_battery(
            current_metrics=metrics,
            previous_metrics=previous_metrics,
            alpha=alpha,
        )

    # Record current run in history (preserving raw score arrays)
    current_run = {
        "target_id": target_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "summary": _flatten_metrics_summary(metrics),
        "metrics": metrics,
        "regression_detected": regression_report["has_regression"] if regression_report else False,
    }

    history.append(current_run)
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Maintain human-readable CSV summary
    csv_row = dict(current_run["summary"])
    csv_row["regression_detected"] = current_run["regression_detected"]

    file_exists = performance_csv_path.exists()
    with open(performance_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_row)

    # Calculate scalar differences for quick inspection
    diff = None
    if previous_run:
        diff = {}
        prev_summary = previous_run.get("summary", {})
        for key, value in current_run["summary"].items():
            if key == "timestamp" or value is None:
                continue
            prev_val = prev_summary.get(key)
            if prev_val is not None:
                diff[key] = round(float(value) - float(prev_val), 4)

    return {
        "target_id": target_id,
        "current": current_run["summary"],
        "previous": previous_run.get("summary") if previous_run else None,
        "diff": diff,
        "regression_report": regression_report,
        "raw_current_metrics": metrics,
        "history_file": str(history_json_path),
        "csv_file": str(performance_csv_path),
    }
