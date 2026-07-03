# use_cases/dp/utils/privacy_scoring.py
"""Centralized helpers for computing privacy scores from structured metrics."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

IMS_THRESHOLD = 0.01  # 1% identical matches
DCR_MIN_BUFFER = 50.0  # absolute safety margin for DCR
DCR_MULTIPLE = 5.0  # minimum multiplier over train distance
NNDR_LOW = 0.2  # values below this considered risky
NNDR_HIGH = 0.9  # values at/above this considered safe


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _score_ims(ims_value: Optional[float]) -> Optional[float]:
    if ims_value is None:
        return None
    score = _clamp(1.0 - ims_value / IMS_THRESHOLD) * 100.0
    return score


def _score_dcr(syn: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if syn is None:
        return None
    base = baseline if baseline and baseline > 0 else 0.0
    safe_target = max(base + DCR_MIN_BUFFER, base * DCR_MULTIPLE, DCR_MIN_BUFFER)
    if syn <= base:
        return 0.0
    if safe_target <= base:
        return 100.0
    ratio = _clamp((syn - base) / (safe_target - base))
    return ratio * 100.0


def _score_nndr(nndr_value: Optional[float]) -> Optional[float]:
    if nndr_value is None:
        return None
    ratio = _clamp((nndr_value - NNDR_LOW) / (NNDR_HIGH - NNDR_LOW))
    return ratio * 100.0


def _score_anonymeter(risk: Optional[float]) -> Optional[float]:
    if risk is None:
        return None
    return _clamp(1.0 - float(risk)) * 100.0


def compute_privacy_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute an aggregated privacy summary using non-MIA metrics."""
    summary: Dict[str, Any] = {
        "available": False,
        "score": None,
        "level": "Unknown",
        "color": "#6c757d",
        "subtitle": "Structured privacy metrics (IMS, DCR, NNDR)",
        "components": [],
        "notes": [],
        "message": None,
        "exact_matches": None,
        "structured_metrics": None,
        "anonymeter": None,
    }

    if not isinstance(results, dict):
        summary["message"] = "Results payload missing."
        return summary

    privacy_block = results.get("privacy")
    if not isinstance(privacy_block, dict):
        summary["message"] = "Privacy metrics not captured for this run."
        return summary

    structured = privacy_block.get("structured_privacy_metrics")
    if not isinstance(structured, dict):
        structured = {}

    summary["exact_matches"] = (
        privacy_block.get("exact_matches") if isinstance(privacy_block, dict) else None
    )
    summary["structured_metrics"] = structured
    anonymeter = privacy_block.get("anonymeter")
    summary["anonymeter"] = anonymeter if isinstance(anonymeter, dict) else None

    components: List[float] = []
    component_details: List[Dict[str, Any]] = []
    notes: List[str] = []

    # IMS / Exact matches
    ims_metric = structured.get("IMS")
    ims_value = None
    if isinstance(ims_metric, dict):
        ims_value = _to_float(ims_metric.get("ims_syn_train"))
    if ims_value is None:
        exact_matches = privacy_block.get("exact_matches")
        if isinstance(exact_matches, dict):
            ims_value = _to_float(exact_matches.get("exact_match_percentage"))
    ims_score = _score_ims(ims_value)
    if ims_score is not None:
        if ims_value is not None and ims_value >= IMS_THRESHOLD:
            notes.append(
                f"Identical match share ({ims_value:.4f}) exceeds the {IMS_THRESHOLD:.4f} threshold."
            )
        component_details.append(
            {
                "name": "IMS",
                "label": "Identical Match Share",
                "score": ims_score,
                "value": ims_value,
                "threshold": IMS_THRESHOLD,
                "direction": "lower_better",
                "description": "Proportion of synthetic rows that exactly match training records.",
            }
        )
        components.append(ims_score)

    # DCR
    dcr_metric = structured.get("DCR")
    syn_dcr = (
        _to_float(dcr_metric.get("syn_train_5pct"))
        if isinstance(dcr_metric, dict)
        else None
    )
    train_dcr = (
        _to_float(dcr_metric.get("train_train_5pct"))
        if isinstance(dcr_metric, dict)
        else None
    )
    dcr_score = _score_dcr(syn_dcr, train_dcr)
    if dcr_score is not None:
        if syn_dcr is not None and train_dcr is not None and syn_dcr <= train_dcr:
            notes.append(
                "Synthetic records are as close or closer to real data than the training baseline (DCR)."
            )
        component_details.append(
            {
                "name": "DCR",
                "label": "Distance to Closest Record",
                "score": dcr_score,
                "value": syn_dcr,
                "baseline": train_dcr,
                "target": max(
                    train_dcr or 0.0 + DCR_MIN_BUFFER,
                    (train_dcr or 0.0) * DCR_MULTIPLE,
                    DCR_MIN_BUFFER,
                ),
                "direction": "higher_better",
                "description": "5th percentile distance between synthetic rows and their nearest real neighbour.",
            }
        )
        components.append(dcr_score)

    # NNDR
    nndr_metric = structured.get("NNDR")
    syn_nndr = (
        _to_float(nndr_metric.get("syn_train_5pct"))
        if isinstance(nndr_metric, dict)
        else None
    )
    nndr_score = _score_nndr(syn_nndr)
    if nndr_score is not None:
        if syn_nndr is not None and syn_nndr < NNDR_LOW:
            notes.append(
                f"Nearest-neighbour distance ratio ({syn_nndr:.3f}) is very low, indicating potential memorisation."
            )
        component_details.append(
            {
                "name": "NNDR",
                "label": "Nearest-Neighbour Distance Ratio",
                "score": nndr_score,
                "value": syn_nndr,
                "target": NNDR_HIGH,
                "direction": "higher_better",
                "description": "Compares nearest-neighbour distance with average distance; values near 1.0 are safer.",
            }
        )
        components.append(nndr_score)

    # Anonymeter (optional, may include multiple sub-metrics)
    if isinstance(anonymeter, dict):
        for metric_name, metric_block in anonymeter.items():
            if isinstance(metric_block, dict):
                risk = _to_float(metric_block.get("risk"))
                score = _score_anonymeter(risk)
                if score is None:
                    continue
                component_details.append(
                    {
                        "name": f"Anonymeter:{metric_name}",
                        "label": metric_block.get("label", metric_name),
                        "score": score,
                        "value": risk,
                        "direction": "lower_better",
                        "description": metric_block.get("desc")
                        or metric_block.get("description"),
                    }
                )
                components.append(score)
                if risk is not None and risk > 0.5:
                    notes.append(
                        f"Anonymeter risk '{metric_name}' is elevated ({risk:.2f})."
                    )

    if not components:
        summary["message"] = "No structured privacy metrics available to score."
        summary["notes"] = notes
        return summary

    final_score = sum(components) / len(components)
    summary.update(
        {
            "available": True,
            "score": final_score,
            "components": component_details,
            "notes": notes,
        }
    )

    if final_score >= 80:
        summary["level"] = "High"
        summary["color"] = "#28a745"
        summary["message"] = (
            summary.get("message")
            or "Strong privacy protections detected across structured metrics."
        )
    elif final_score >= 60:
        summary["level"] = "Medium"
        summary["color"] = "#ffc107"
        summary["message"] = (
            summary.get("message")
            or "Privacy posture is acceptable but warrants monitoring."
        )
    else:
        summary["level"] = "Low"
        summary["color"] = "#dc3545"
        summary["message"] = (
            summary.get("message")
            or "Structured privacy metrics indicate elevated disclosure risk."
        )

    return summary
