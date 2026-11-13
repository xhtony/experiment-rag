"""Shared evaluation contract, independent of model calls and persistence."""

import math

EVALUATION_PROTOCOL = "phase2-evaluation-v3"
STAGE_METRICS = {
    "response": ("answered", "answerable_coverage"),
    "evidence_coverage": ("citation_validity", "gold_citation_overlap_precision",
                          "gold_citation_overlap_recall", "retrieval_recall", "candidate_recall"),
    "claims": ("total_claims",),
    "reference_grounding": ("supported_claims", "partially_supported_claims", "unsupported_claims",
                            "contradicted_claims", "reference_groundedness", "reference_weighted_support",
                            "hallucination_rate", "hallucinated"),
    "retrieved_context_support": ("retrieved_context_faithfulness",),
    "citation_links": ("citation_precision",),
    "citation_completeness": ("citation_completeness",),
    "behaviour": ("action_appropriate", "appropriate_refusal", "false_refusal",
                  "appropriate_clarification", "unnecessary_clarification", "inappropriate_answer"),
}
METRIC_STAGE = {metric: stage for stage, metrics in STAGE_METRICS.items() for metric in metrics}
JUDGED_STAGES = set(STAGE_METRICS) - {"response", "evidence_coverage"}
TERMINAL_OK = {"success", "not_applicable"}
METRIC_STATUSES = {"ok", "not_applicable", "evaluation_failed", "blocked", "pending"}


def evaluation_status(metrics, details):
    """A row is not complete merely because it exists or says 'complete'."""
    state = (details or {}).get("evaluation", {})
    if state.get("protocol") != EVALUATION_PROTOCOL:
        return "legacy_unversioned"
    stages, states = state.get("stages", {}), state.get("metric_states", {})
    if set(stages) != set(STAGE_METRICS) or set(states) != set(METRIC_STAGE) or set(metrics) != set(METRIC_STAGE):
        return "invalid"
    for name, meta in states.items():
        status = meta.get("status")
        if status not in METRIC_STATUSES or meta.get("stage") != METRIC_STAGE[name]:
            return "invalid"
        value = metrics[name]
        if status == "ok":
            if not isinstance(value, (int, float, bool)) or not math.isfinite(value):
                return "invalid"
        elif value is not None or not meta.get("reason"):
            return "invalid"
        source_status = stages[METRIC_STAGE[name]].get("status")
        if status == "ok" and source_status != "success":
            return "invalid"
        if status == "not_applicable" and source_status not in TERMINAL_OK:
            return "invalid"
    if all(s.get("status") in TERMINAL_OK for s in stages.values()) and all(
        m["status"] in {"ok", "not_applicable"} for m in states.values()
    ):
        return "complete"
    if any(stages[s].get("status") == "success" for s in JUDGED_STAGES):
        return "partial"
    if any(s.get("status") in {"failed", "blocked"} for s in stages.values()):
        return "failed"
    return "pending"
