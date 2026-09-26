import math
from typing import List, Dict, Any, Tuple, Optional
import scipy.stats as stats


def cliffs_delta(after: List[float], before: List[float]) -> Tuple[float, str]:
    """
    Computes Cliff's Delta effect size between 'after' and 'before' scores.
    Formula:
        delta = (# pairs where after > before - # pairs where after < before) / (n_before * n_after)

    Ranges from -1.0 to 1.0.
    Standard Romano et al. (2006) thresholds:
        |delta| < 0.147 -> "negligible"
        |delta| < 0.330 -> "small"
        |delta| < 0.474 -> "medium"
        |delta| >= 0.474 -> "large"
    """
    n_after = len(after)
    n_before = len(before)
    if n_after == 0 or n_before == 0:
        return 0.0, "negligible"

    more = 0
    less = 0
    for a in after:
        for b in before:
            if a > b:
                more += 1
            elif a < b:
                less += 1

    delta = (more - less) / (n_after * n_before)
    abs_delta = abs(delta)

    if abs_delta < 0.147:
        size = "negligible"
    elif abs_delta < 0.330:
        size = "small"
    elif abs_delta < 0.474:
        size = "medium"
    else:
        size = "large"

    return round(delta, 4), size


def mann_whitney_u_test(after: List[float], before: List[float]) -> float:
    """
    Mann-Whitney U test with alternative='less' (one-sided drop test: after < before).
    Returns the p-value.
    """
    if len(after) == 0 or len(before) == 0:
        return 1.0

    # If all values are identical across both groups, p-value is 1.0 (no difference)
    if len(set(after + before)) <= 1:
        return 1.0

    try:
        res = stats.mannwhitneyu(after, before, alternative="less")
        return float(res.pvalue)
    except Exception:
        return 1.0


def fishers_exact_test(
    after_successes: List[bool], before_successes: List[bool]
) -> Tuple[float, float, float]:
    """
    Fisher's exact test for 2x2 contingency table:
        table = [[after_pass, after_fail],
                 [before_pass, before_fail]]
    Alternative hypothesis is 'less' (testing if after pass rate < before pass rate).

    Returns:
        (p_value, after_pass_rate, before_pass_rate)
    """
    n_after = len(after_successes)
    n_before = len(before_successes)
    if n_after == 0 or n_before == 0:
        return 1.0, 0.0, 0.0

    after_pass = sum(1 for s in after_successes if s)
    after_fail = n_after - after_pass

    before_pass = sum(1 for s in before_successes if s)
    before_fail = n_before - before_pass

    after_rate = round(after_pass / n_after, 4)
    before_rate = round(before_pass / n_before, 4)

    table = [
        [after_pass, after_fail],
        [before_pass, before_fail],
    ]

    try:
        _, p_val = stats.fisher_exact(table, alternative="less")
        return float(p_val), after_rate, before_rate
    except Exception:
        return 1.0, after_rate, before_rate


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """
    Benjamini-Hochberg (FDR) p-value correction for multiple comparisons.
    Ensures false-alarm rates don't stack up across multiple metric & slice tests.
    """
    m = len(p_values)
    if m <= 1:
        return [p for p in p_values]

    # Keep track of original indices
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    # Calculate BH adjusted values
    for rank, (orig_idx, p_val) in enumerate(indexed_p, start=1):
        q = p_val * m / rank
        adjusted[rank - 1] = min(1.0, q)

    # Monotonicity adjustment (from right to left)
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Map back to original order
    result = [0.0] * m
    for (orig_idx, _), adj_p in zip(indexed_p, adjusted):
        result[orig_idx] = round(adj_p, 5)

    return result


def evaluate_regression_battery(
    current_metrics: Dict[str, List[Dict[str, Any]]],
    previous_metrics: Dict[str, List[Dict[str, Any]]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Runs full regression evaluation across all metric categories:
    1. Continuous score drop test: Mann-Whitney U + Cliff's Delta.
    2. Binary pass/fail test: Fisher's Exact Test.
    3. Multi-testing FDR correction: Benjamini-Hochberg across all tests in run.

    Gate rules:
      - Continuous: regression if p_adj < alpha AND delta <= -0.147
      - Binary: regression if p_adj < alpha AND after_pass_rate < before_pass_rate
    """
    metric_keys = [k for k in current_metrics.keys() if k in previous_metrics]

    test_records = []
    raw_p_values = []

    for key in metric_keys:
        curr_entries = current_metrics.get(key, [])
        prev_entries = previous_metrics.get(key, [])

        curr_scores = [float(e["score"]) for e in curr_entries if e.get("score") is not None]
        prev_scores = [float(e["score"]) for e in prev_entries if e.get("score") is not None]

        curr_success = [bool(e["success"]) for e in curr_entries if e.get("success") is not None]
        prev_success = [bool(e["success"]) for e in prev_entries if e.get("success") is not None]

        # 1. Continuous Score Evaluation
        if curr_scores and prev_scores:
            mwu_p = mann_whitney_u_test(after=curr_scores, before=prev_scores)
            delta, effect_size = cliffs_delta(after=curr_scores, before=prev_scores)
            curr_mean = round(sum(curr_scores) / len(curr_scores), 4)
            prev_mean = round(sum(prev_scores) / len(prev_scores), 4)

            test_records.append({
                "metric": key,
                "type": "continuous",
                "test": "Mann-Whitney U + Cliff's Delta",
                "current_mean": curr_mean,
                "previous_mean": prev_mean,
                "effect_size": delta,
                "effect_size_magnitude": effect_size,
                "raw_p_value": mwu_p,
            })
            raw_p_values.append(mwu_p)

        # 2. Binary Success Evaluation
        if curr_success and prev_success:
            fisher_p, curr_rate, prev_rate = fishers_exact_test(
                after_successes=curr_success, before_successes=prev_success
            )

            test_records.append({
                "metric": key,
                "type": "binary",
                "test": "Fisher's Exact Test",
                "current_pass_rate": curr_rate,
                "previous_pass_rate": prev_rate,
                "effect_size": round(curr_rate - prev_rate, 4),
                "effect_size_magnitude": "drop" if curr_rate < prev_rate else "gain_or_equal",
                "raw_p_value": fisher_p,
            })
            raw_p_values.append(fisher_p)

    # 3. Benjamini-Hochberg FDR Correction
    adjusted_p_values = benjamini_hochberg(raw_p_values)

    has_regression = False
    metric_results = {}

    for record, adj_p in zip(test_records, adjusted_p_values):
        record["adjusted_p_value"] = adj_p
        m_type = record["type"]

        if m_type == "continuous":
            # Regression if p < alpha AND non-negligible drop (delta <= -0.147)
            is_reg = bool(adj_p < alpha and record["effect_size"] <= -0.147)
            reason = (
                f"Significant continuous score drop (p_adj={adj_p} < {alpha}, "
                f"Cliff's delta={record['effect_size']} <= -0.147)"
                if is_reg
                else "No statistically significant continuous regression"
            )
        else:
            # Binary regression if p < alpha AND pass rate dropped
            is_reg = bool(adj_p < alpha and record["current_pass_rate"] < record["previous_pass_rate"])
            reason = (
                f"Significant pass rate drop (p_adj={adj_p} < {alpha}, "
                f"{record['previous_pass_rate']} -> {record['current_pass_rate']})"
                if is_reg
                else "No statistically significant binary regression"
            )

        record["is_regression"] = is_reg
        record["reason"] = reason

        if is_reg:
            has_regression = True

        metric_results[f"{record['metric']}_{record['type']}"] = record

    return {
        "has_regression": has_regression,
        "alpha": alpha,
        "total_tests_conducted": len(test_records),
        "results": metric_results,
    }
