from __future__ import annotations

from typing import Any

from .common import require


def evaluate_agent_candidate(cases: list[dict[str, Any]], outputs: list[dict[str, Any]],
                             *, min_case_pass_rate: float = 0.95,
                             min_tool_success_rate: float = 0.98,
                             min_citation_coverage: float = 0.95,
                             max_critical_failures: int = 0) -> dict[str, Any]:
    """Evaluate recorded Agent outputs against explicit replay expectations.

    The caller runs the candidate Agent on a frozen replay set and passes the
    actual outputs here. This module never asks an LLM to grade itself.
    """
    require(len(cases) > 0 and len(cases) == len(outputs),
            "cases and outputs must have the same nonzero length")
    for name, value in (("min_case_pass_rate", min_case_pass_rate),
                        ("min_tool_success_rate", min_tool_success_rate),
                        ("min_citation_coverage", min_citation_coverage)):
        require(0 <= value <= 1, f"{name} must be in [0,1]")
    require(type(max_critical_failures) is int and max_critical_failures >= 0,
            "max_critical_failures must be nonnegative")

    case_results: list[dict[str, Any]] = []
    tool_total = tool_success = citation_required = citation_present = 0
    critical_failures = 0
    for case, output in zip(cases, outputs):
        case_id = str(case.get("case_id", "unknown"))
        failures: list[str] = []
        allowed_decisions = case.get("allowed_decisions", [])
        decision = output.get("decision")
        if not allowed_decisions or decision not in allowed_decisions:
            failures.append("decision_not_allowed")

        expected_manual = bool(case.get("manual_review_expected", False))
        actual_manual = bool(output.get("manual_review_requested", False))
        if expected_manual != actual_manual:
            failures.append("manual_review_mismatch")

        actual_tools = output.get("tool_calls", [])
        actual_tool_names = {str(call.get("name")) for call in actual_tools if isinstance(call, dict)}
        missing_tools = sorted(set(case.get("required_tools", [])) - actual_tool_names)
        if missing_tools:
            failures.append("required_tools_missing:" + ",".join(missing_tools))
        for call in actual_tools:
            if isinstance(call, dict):
                tool_total += 1
                if call.get("success") is True:
                    tool_success += 1
                else:
                    failures.append("tool_call_failed")

        if case.get("citations_required", False):
            citation_required += 1
            citations = output.get("citations", [])
            valid_citations = [c for c in citations if isinstance(c, dict) and
                               c.get("source_id") and c.get("source_version")]
            if valid_citations:
                citation_present += 1
            else:
                failures.append("required_citation_missing_or_unversioned")

        passed = not failures
        if bool(case.get("critical", False)) and not passed:
            critical_failures += 1
        case_results.append({"case_id": case_id, "passed": passed,
                             "critical": bool(case.get("critical", False)),
                             "failures": failures})

    pass_rate = sum(x["passed"] for x in case_results) / len(case_results)
    tool_rate = tool_success / tool_total if tool_total else 1.0
    citation_rate = citation_present / citation_required if citation_required else 1.0
    failed_reasons = []
    if pass_rate < min_case_pass_rate:
        failed_reasons.append("case_pass_rate_below_minimum")
    if tool_rate < min_tool_success_rate:
        failed_reasons.append("tool_success_rate_below_minimum")
    if citation_rate < min_citation_coverage:
        failed_reasons.append("citation_coverage_below_minimum")
    if critical_failures > max_critical_failures:
        failed_reasons.append("critical_case_failure")
    return {
        "status": "PASS" if not failed_reasons else "REJECT",
        "allow_upgrade": not failed_reasons,
        "metrics": {"case_pass_rate": pass_rate, "tool_success_rate": tool_rate,
                    "citation_coverage": citation_rate, "critical_failures": critical_failures,
                    "case_count": len(cases), "tool_call_count": tool_total,
                    "citation_required_cases": citation_required},
        "thresholds": {"min_case_pass_rate": min_case_pass_rate,
                       "min_tool_success_rate": min_tool_success_rate,
                       "min_citation_coverage": min_citation_coverage,
                       "max_critical_failures": max_critical_failures},
        "failed_reasons": failed_reasons,
        "cases": case_results,
        "note": "Replay expectations must be authored/reviewed by humans and frozen before candidate evaluation.",
    }


def compare_agent_candidates(cases: list[dict[str, Any]],
                             incumbent_outputs: list[dict[str, Any]],
                             candidate_outputs: list[dict[str, Any]], *,
                             max_case_pass_rate_drop: float = 0.0,
                             max_tool_success_rate_drop: float = 0.0,
                             max_citation_coverage_drop: float = 0.0,
                             **candidate_thresholds: Any) -> dict[str, Any]:
    """Compare incumbent and candidate on exactly the same frozen replay cases."""
    incumbent = evaluate_agent_candidate(cases, incumbent_outputs, **candidate_thresholds)
    candidate = evaluate_agent_candidate(cases, candidate_outputs, **candidate_thresholds)
    for name, value in (("max_case_pass_rate_drop", max_case_pass_rate_drop),
                        ("max_tool_success_rate_drop", max_tool_success_rate_drop),
                        ("max_citation_coverage_drop", max_citation_coverage_drop)):
        require(0 <= value <= 1, f"{name} must be in [0,1]")
    regressions = []
    for old_case, new_case in zip(incumbent["cases"], candidate["cases"]):
        if old_case["passed"] and not new_case["passed"]:
            regressions.append({"case_id": new_case["case_id"],
                                "critical": new_case["critical"],
                                "failures": new_case["failures"]})
    old = incumbent["metrics"]
    new = candidate["metrics"]
    comparisons = {
        "case_pass_rate": {"incumbent": old["case_pass_rate"],
                            "candidate": new["case_pass_rate"],
                            "passed": new["case_pass_rate"] >= old["case_pass_rate"] - max_case_pass_rate_drop},
        "tool_success_rate": {"incumbent": old["tool_success_rate"],
                               "candidate": new["tool_success_rate"],
                               "passed": new["tool_success_rate"] >= old["tool_success_rate"] - max_tool_success_rate_drop},
        "citation_coverage": {"incumbent": old["citation_coverage"],
                              "candidate": new["citation_coverage"],
                              "passed": new["citation_coverage"] >= old["citation_coverage"] - max_citation_coverage_drop},
    }
    reasons = []
    if not candidate["allow_upgrade"]:
        reasons.append("candidate_did_not_meet_agent_gate_thresholds")
    if any(not value["passed"] for value in comparisons.values()):
        reasons.append("candidate_regressed_against_incumbent")
    if any(item["critical"] for item in regressions):
        reasons.append("critical_replay_regression")
    return {
        "status": "PASS" if not reasons else "REJECT",
        "allow_upgrade": not reasons,
        "incumbent": incumbent,
        "candidate": candidate,
        "comparisons": comparisons,
        "new_regressions": regressions,
        "failed_reasons": reasons,
        "note": "Both versions must run on the same frozen cases; the candidate is not self-graded.",
    }
