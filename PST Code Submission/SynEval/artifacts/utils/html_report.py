"""HTML dashboard generator for SynEval evaluation results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DimensionSummary:
    name: str
    score_display: str
    score_percent: float
    description: str
    highlights: List[str] = field(default_factory=list)
    status: str = "ok"  # ok | warning | error | skipped


class EvaluationHTMLGenerator:
    """Generate a lightweight HTML dashboard summarizing SynEval metrics."""

    def __init__(
        self,
        output_path: Path | str = "artifacts/html/syneval_dashboard.html",
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: Dict[str, Any],
        command: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        summaries = self._build_summaries(results)
        html = self._render_html(summaries, results, command, context)
        self.output_path.write_text(html, encoding="utf-8")
        return self.output_path

    # ------------------------------------------------------------------ helpers
    def _build_summaries(self, results: Dict[str, Any]) -> List[DimensionSummary]:
        summaries: List[DimensionSummary] = []

        summaries.append(self._summarize_fidelity(results.get("fidelity")))
        summaries.append(self._summarize_utility(results.get("utility")))
        summaries.append(self._summarize_diversity(results.get("diversity")))
        summaries.append(self._summarize_privacy(results.get("privacy")))

        return [summary for summary in summaries if summary is not None]

    def _summarize_fidelity(self, fidelity: Any) -> Optional[DimensionSummary]:
        if fidelity is None:
            return None

        if isinstance(fidelity, dict) and fidelity.get("error"):
            return DimensionSummary(
                name="Fidelity",
                score_display="Error",
                score_percent=0,
                description="Statistical fidelity evaluation failed.",
                highlights=[str(fidelity.get("error"))],
                status="poor",
            )

        if isinstance(fidelity, dict) and fidelity.get("skipped"):
            reason = fidelity.get("reason", "Skipped at user request")
            return DimensionSummary(
                name="Fidelity",
                score_display="Skipped",
                score_percent=0,
                description="Fidelity evaluation was skipped.",
                highlights=[reason],
                status="skipped",
            )

        diag = fidelity.get("diagnostic", {}) if isinstance(fidelity, dict) else {}
        overall = diag.get("Overall", {}) if isinstance(diag, dict) else {}
        score = self._as_float(overall.get("score"))
        score_percent = self._normalize_score(score)
        data_validity = self._format_percentage(diag.get("Data Validity"))
        data_structure = self._format_percentage(diag.get("Data Structure"))

        highlights: List[str] = []
        if data_validity:
            highlights.append(f"Data Validity: {data_validity}")
        if data_structure:
            highlights.append(f"Data Structure: {data_structure}")

        quality = fidelity.get("quality", {}) if isinstance(fidelity, dict) else {}
        quality_score = self._format_percentage(
            (quality.get("Overall") or {}).get("score") if isinstance(quality, dict) else None
        )
        if quality_score:
            highlights.append(f"Quality Score: {quality_score}")

        status_label, _ = self._get_status_from_score(score_percent)
        status = status_label.lower()

        return DimensionSummary(
            name="Fidelity",
            score_display=self._format_percentage(score),
            score_percent=score_percent,
            description="Measures how well synthetic data preserves statistical characteristics.",
            highlights=highlights,
            status=status,
        )

    def _summarize_utility(self, utility: Any) -> Optional[DimensionSummary]:
        if utility is None:
            return None

        if isinstance(utility, dict) and utility.get("error"):
            return DimensionSummary(
                name="Utility",
                score_display="Error",
                score_percent=0,
                description="Utility evaluation failed.",
                highlights=[str(utility.get("error"))],
                status="poor",
            )

        if isinstance(utility, dict) and utility.get("skipped"):
            reason = utility.get("reason", "Skipped at user request")
            return DimensionSummary(
                name="Utility",
                score_display="Skipped",
                score_percent=0,
                description="Utility evaluation was skipped.",
                highlights=[reason],
                status="skipped",
            )

        highlights: List[str] = []
        score = None

        tstr = utility.get("tstr_accuracy") if isinstance(utility, dict) else None
        if isinstance(tstr, dict):
            real_model = tstr.get("real_data_model", {})
            syn_model = tstr.get("synthetic_data_model", {})
            real_acc = self._as_float(real_model.get("accuracy"))
            syn_acc = self._as_float(syn_model.get("accuracy"))
            ratio = self._safe_ratio(syn_acc, real_acc)
            score = ratio
            if real_acc is not None:
                highlights.append(f"Real Model Accuracy: {self._format_percentage(real_acc)}")
            if syn_acc is not None:
                highlights.append(f"Synthetic Model Accuracy: {self._format_percentage(syn_acc)}")
            if ratio is not None:
                ratio_display = self._format_ratio(ratio)
                if ratio_display:
                    highlights.append(f"Performance Ratio (Syn/Real): {ratio_display}")
            task_type = tstr.get("task_type")
            if task_type:
                highlights.append(f"Task Type: {task_type}")
        else:
            real_model = utility.get("real_data_model", {})
            syn_model = utility.get("synthetic_data_model", {})
            real_acc = self._as_float(real_model.get("accuracy"))
            syn_acc = self._as_float(syn_model.get("accuracy"))
            ratio = self._safe_ratio(syn_acc, real_acc)
            score = ratio
            if real_acc is not None:
                highlights.append(f"Real Model Accuracy: {self._format_percentage(real_acc)}")
            if syn_acc is not None:
                highlights.append(f"Synthetic Model Accuracy: {self._format_percentage(syn_acc)}")
            if ratio is not None:
                ratio_display = self._format_ratio(ratio)
                if ratio_display:
                    highlights.append(f"Performance Ratio (Syn/Real): {ratio_display}")

        score_percent = self._normalize_score(score)
        score_display = self._format_percentage(score) or "N/A"

        status_label, _ = self._get_status_from_score(score_percent)
        status = status_label.lower()

        return DimensionSummary(
            name="Utility",
            score_display=score_display,
            score_percent=score_percent,
            description="Evaluates how well models trained on synthetic data generalize to real data (TSTR).",
            highlights=highlights or ["Utility metrics available"],
            status=status,
        )

    def _summarize_diversity(self, diversity: Any) -> Optional[DimensionSummary]:
        if diversity is None:
            return None

        if isinstance(diversity, dict) and diversity.get("error"):
            return DimensionSummary(
                name="Diversity",
                score_display="Error",
                score_percent=0,
                description="Diversity evaluation failed.",
                highlights=[str(diversity.get("error"))],
                status="poor",
            )

        if isinstance(diversity, dict) and diversity.get("skipped"):
            reason = diversity.get("reason", "Skipped at user request")
            return DimensionSummary(
                name="Diversity",
                score_display="Skipped",
                score_percent=0,
                description="Diversity evaluation was skipped.",
                highlights=[reason],
                status="skipped",
            )

        tabular = diversity.get("tabular_diversity", {}) if isinstance(diversity, dict) else {}
        text_diversity = diversity.get("text_diversity", {}) if isinstance(diversity, dict) else {}
        
        # Get coverage from tabular data
        tabular_coverage = tabular.get("coverage", {}) if isinstance(tabular, dict) else {}
        # Get coverage from text data
        text_coverage = text_diversity.get("coverage", {}) if isinstance(text_diversity, dict) else {}
        
        # Merge both coverage dictionaries
        coverage = {**tabular_coverage, **text_coverage}
        
        uniqueness = tabular.get("uniqueness", {}) if isinstance(tabular, dict) else {}
        entropy = (
            (tabular.get("entropy_metrics") or {}).get("dataset_entropy", {})
            if isinstance(tabular, dict)
            else {}
        )

        avg_coverage = None
        if isinstance(coverage, dict) and coverage:
            # Exclude ID/PII columns (typically have 0% coverage and are not meaningful for diversity)
            # Also exclude columns with very low coverage (< 1%) as they're likely ID columns
            numeric_values = []
            for col_name, value in coverage.items():
                val = self._as_float(value)
                if val is not None:
                    # Exclude columns with 0% coverage (likely ID columns)
                    # Keep columns with meaningful coverage (> 0%)
                    if val > 0:
                        numeric_values.append(val)
            
            if numeric_values:
                avg_coverage = sum(numeric_values) / len(numeric_values)
            elif coverage:
                # If all columns have 0% coverage, still calculate average for transparency
                all_values = [self._as_float(v) for v in coverage.values()]
                all_values = [v for v in all_values if v is not None]
                if all_values:
                    avg_coverage = sum(all_values) / len(all_values)

        duplicate_ratio = self._as_float(uniqueness.get("synthetic_duplicate_ratio"))
        entropy_ratio = self._as_float(entropy.get("entropy_ratio"))
        
        # Get text diversity metrics for additional context
        text_diversity_data = text_diversity if isinstance(text_diversity, dict) else {}
        text_unique_ratios = []
        if isinstance(text_diversity_data, dict):
            for col_name, col_data in text_diversity_data.get("synthetic", {}).items():
                if isinstance(col_data, dict):
                    lexical = col_data.get("lexical_diversity", {})
                    # Use 1-gram unique ratio as a diversity indicator
                    if "1-gram" in lexical and isinstance(lexical["1-gram"], dict):
                        unique_ratio = self._as_float(lexical["1-gram"].get("unique_ratio"))
                        if unique_ratio is not None:
                            text_unique_ratios.append(unique_ratio)

        # Calculate diversity score based on entropy ratio and uniqueness
        # Diversity should measure: 1) Information richness (entropy), 2) Uniqueness (low duplication)
        diversity_score = None
        
        if entropy_ratio is not None:
            # Entropy ratio: ideal is 1.0 (100%), meaning synthetic data has similar diversity to original
            # Score: 100% - |1.0 - entropy_ratio| * 100, capped at 0-100%
            # If entropy_ratio is 0.9-1.1, score is 90-100%
            # If entropy_ratio is 0.8-1.2, score is 80-100%
            entropy_score = max(0, min(100, 100 - abs(1.0 - entropy_ratio) * 100))
            
            # Uniqueness score: lower duplicate ratio is better
            # Score: 100% - duplicate_ratio (if duplicate_ratio is 0-100%)
            uniqueness_score = None
            if duplicate_ratio is not None:
                uniqueness_score = max(0, min(100, 100 - duplicate_ratio))
            
            # Combine scores: 70% entropy + 30% uniqueness (if available)
            if uniqueness_score is not None:
                diversity_score = 0.7 * entropy_score + 0.3 * uniqueness_score
            else:
                diversity_score = entropy_score
        
        # Fallback: if no entropy ratio, use uniqueness only
        elif duplicate_ratio is not None:
            uniqueness_score = max(0, min(100, 100 - duplicate_ratio))
            diversity_score = uniqueness_score

        highlights: List[str] = []
        if entropy_ratio is not None:
            entropy_ratio_display = self._format_ratio(entropy_ratio)
            if entropy_ratio_display:
                highlights.append(f"Entropy Ratio (Syn/Real): {entropy_ratio_display}")
        if duplicate_ratio is not None:
            highlights.append(f"Synthetic Duplicate Ratio: {duplicate_ratio:.2f}%")
        if avg_coverage is not None:
            # Coverage is shown as reference, not as the main score
            if isinstance(coverage, dict) and coverage:
                all_values = [self._as_float(v) for v in coverage.values()]
                all_values = [v for v in all_values if v is not None]
                if all_values:
                    overall_avg = sum(all_values) / len(all_values)
                    highlights.append(f"Value Coverage: {overall_avg:.1f}%")
        if text_unique_ratios:
            avg_text_unique = sum(text_unique_ratios) / len(text_unique_ratios) * 100
            highlights.append(f"Text Vocabulary Diversity: {avg_text_unique:.1f}%")

        score_percent = self._normalize_score(
            diversity_score / 100 if isinstance(diversity_score, (int, float)) else None
        )
        score_display = (
            f"{(diversity_score or 0):.1f}%" if diversity_score is not None else "N/A"
        )

        status_label, _ = self._get_status_from_score(score_percent)
        status = status_label.lower()

        return DimensionSummary(
            name="Diversity",
            score_display=score_display,
            score_percent=score_percent,
            description="Assesses variety, uniqueness, and entropy of synthetic data.",
            highlights=highlights or ["Diversity metrics available"],
            status=status,
        )

    def _summarize_privacy(self, privacy: Any) -> Optional[DimensionSummary]:
        if privacy is None:
            return None

        if isinstance(privacy, dict) and privacy.get("error"):
            return DimensionSummary(
                name="Privacy",
                score_display="Error",
                score_percent=0,
                description="Privacy evaluation failed.",
                highlights=[str(privacy.get("error"))],
                status="poor",
            )

        if isinstance(privacy, dict) and privacy.get("skipped"):
            reason = privacy.get("reason", "Skipped at user request")
            return DimensionSummary(
                name="Privacy",
                score_display="Skipped",
                score_percent=0,
                description="Privacy evaluation was skipped.",
                highlights=[reason],
                status="skipped",
            )

        highlights: List[str] = []
        
        # Extract all privacy metrics
        mia = privacy.get("membership_inference") if isinstance(privacy, dict) else {}
        exact = privacy.get("exact_matches") if isinstance(privacy, dict) else {}
        structured = privacy.get("structured_privacy_metrics") if isinstance(privacy, dict) else {}
        pii_analysis = privacy.get("pii_analysis") if isinstance(privacy, dict) else {}
        reid_risks = privacy.get("reidentification_risks") if isinstance(privacy, dict) else {}
        
        # Calculate individual privacy scores (0-100, higher is better)
        privacy_scores = []
        score_details = []
        
        # 1. Membership Inference Attack (MIA) Score
        # AUC close to 0.5 = good privacy; close to 1.0 = poor privacy.
        # Score = 100 * (1 - |AUC - 0.5| * 2), where AUC in [0.5, 1.0]
        # If AUC = 0.5, score = 100; if AUC = 1.0, score = 0
        mia_score = None
        if isinstance(mia, dict):
            auc = self._as_float(
                mia.get("distinguishability_auc")
                or mia.get("mia_auc_score")
                or mia.get("auc")
            )
            if auc is not None:
                # Normalize AUC to [0.5, 1.0] range (AUC < 0.5 means better than random, treat as 0.5)
                normalized_auc = max(0.5, min(1.0, auc))
                # Convert to score: 0.5 -> 100, 1.0 -> 0
                mia_score = 100 * (1 - (normalized_auc - 0.5) * 2)
                privacy_scores.append(mia_score)
                score_details.append(f"MIA: {mia_score:.1f}%")
                
                auc_display = self._format_ratio(auc)
                if auc_display:
                    highlights.append(
                        f"Membership Inference AUC: {auc_display} (lower is better)"
                    )
            syn_conf = self._as_float(mia.get("synthetic_confidence"))
            orig_conf = self._as_float(mia.get("original_confidence"))
            if syn_conf is not None and orig_conf is not None:
                syn_conf_display = self._format_percentage(syn_conf)
                orig_conf_display = self._format_percentage(orig_conf)
                if syn_conf_display and orig_conf_display:
                    highlights.append(
                        f"Classifier Confidence (Syn/Orig): {syn_conf_display} / {orig_conf_display}"
                    )
        
        # 2. Exact Matches Score
        # Lower exact match percentage is better
        # Score = 100 * (1 - min(exact_pct / 5, 1.0)), where 5% is threshold
        exact_score = None
        if isinstance(exact, dict):
            exact_pct = self._as_float(exact.get("exact_match_percentage"))
            if exact_pct is not None:
                # 0% -> 100, 5% -> 0, >5% -> 0
                exact_score = 100 * max(0, 1 - min(exact_pct / 5.0, 1.0))
                privacy_scores.append(exact_score)
                score_details.append(f"Exact Matches: {exact_score:.1f}%")
                
                highlights.append(f"Exact Match Percentage: {exact_pct:.2f}%")
            risk_level = exact.get("risk_level")
            if risk_level:
                highlights.append(f"Exact Match Risk: {risk_level}")
        
        # 3. Structured Privacy Metrics Score (IMS, DCR, NNDR)
        # IMS: lower is better (passed = good)
        # DCR: higher is better (passed = good)
        # NNDR: lower is better (passed = good)
        structured_score = None
        if isinstance(structured, dict):
            structured_scores = []
            if "IMS" in structured:
                ims = structured["IMS"]
                if isinstance(ims, dict) and ims.get("passed"):
                    structured_scores.append(100)
                else:
                    # IMS syn_train should be low, train_test should be high
                    ims_syn = self._as_float(ims.get("ims_syn_train", 1.0))
                    ims_train_test = self._as_float(ims.get("train_test_ims", 0.0))
                    # Lower syn_train and higher train_test is better
                    ims_score = 100 * (1 - ims_syn) * ims_train_test
                    structured_scores.append(max(0, min(100, ims_score)))
            
            if "DCR" in structured:
                dcr = structured["DCR"]
                if isinstance(dcr, dict) and dcr.get("passed"):
                    structured_scores.append(100)
                else:
                    # DCR: syn_train should be much higher than train_train
                    syn_train = self._as_float(dcr.get("syn_train_5pct", 0))
                    train_train = self._as_float(dcr.get("train_train_5pct", 1))
                    if train_train > 0:
                        ratio = syn_train / train_train
                        # Higher ratio is better, but cap at reasonable value
                        dcr_score = 100 * min(1.0, ratio / 10.0)  # Assume 10x is good
                        structured_scores.append(max(0, min(100, dcr_score)))
                    else:
                        structured_scores.append(0)
            
            if "NNDR" in structured:
                nndr = structured["NNDR"]
                if isinstance(nndr, dict) and nndr.get("passed"):
                    structured_scores.append(100)
                else:
                    # NNDR: should be close to train_train value
                    syn_train = self._as_float(nndr.get("syn_train_5pct", 1.0))
                    train_train = self._as_float(nndr.get("train_train_5pct", 1.0))
                    if train_train > 0:
                        ratio = syn_train / train_train
                        # Closer to 1.0 is better
                        nndr_score = 100 * (1 - abs(ratio - 1.0))
                        structured_scores.append(max(0, min(100, nndr_score)))
                    else:
                        structured_scores.append(0)
            
            if structured_scores:
                structured_score = sum(structured_scores) / len(structured_scores)
                privacy_scores.append(structured_score)
                score_details.append(f"Structured Metrics: {structured_score:.1f}%")
        
        # 4. Reidentification Risks Score (Anonymeter)
        # Lower risk is better, risk in [0, 1]
        # Score = 100 * (1 - risk)
        reid_score = None
        if isinstance(reid_risks, dict):
            reid_risks_list = []
            
            # Singling out (univariate and multivariate)
            for key in ["singling_out_univariate", "singling_out_multivariate"]:
                if key in reid_risks:
                    risk = self._as_float(reid_risks[key].get("risk"))
                    if risk is not None:
                        reid_risks_list.append(risk)
            
            # Linkability
            if "linkability" in reid_risks:
                risk = self._as_float(reid_risks["linkability"].get("risk"))
                if risk is not None:
                    reid_risks_list.append(risk)
            
            # Inference (take max risk across all columns)
            if "inference" in reid_risks:
                inference_list = reid_risks["inference"]
                if isinstance(inference_list, list):
                    inference_risks = [
                        self._as_float(item.get("risk"))
                        for item in inference_list
                        if isinstance(item, dict)
                    ]
                    inference_risks = [r for r in inference_risks if r is not None]
                    if inference_risks:
                        reid_risks_list.append(max(inference_risks))
            
            if reid_risks_list:
                # Take maximum risk (worst case)
                max_risk = max(reid_risks_list)
                reid_score = 100 * (1 - min(max_risk, 1.0))
                privacy_scores.append(reid_score)
                score_details.append(f"Reidentification Risks: {reid_score:.1f}%")
        
        # 5. PII Analysis Score
        # Lower exact match percentage is better
        pii_score = None
        if isinstance(pii_analysis, dict):
            pii_scores = []
            for col_name, col_data in pii_analysis.items():
                if isinstance(col_data, dict):
                    exact_pct = self._as_float(col_data.get("exact_match_percentage"))
                    if exact_pct is not None:
                        # 0% -> 100, 5% -> 0
                        col_score = 100 * max(0, 1 - min(exact_pct / 5.0, 1.0))
                        pii_scores.append(col_score)
            
            if pii_scores:
                pii_score = sum(pii_scores) / len(pii_scores)
                privacy_scores.append(pii_score)
                score_details.append(f"PII Analysis: {pii_score:.1f}%")
        
        # Calculate overall privacy score as average of all available scores
        privacy_score = None
        if privacy_scores:
            privacy_score = sum(privacy_scores) / len(privacy_scores)
            # Add score breakdown to highlights
            if len(score_details) > 1:
                highlights.append(f"Score Breakdown: {', '.join(score_details)}")
        
        # Convert to 0-1 range for normalization
        privacy_score_normalized = privacy_score / 100.0 if privacy_score is not None else None
        score_percent = self._normalize_score(privacy_score_normalized)
        score_display = self._format_percentage(privacy_score) if privacy_score is not None else "N/A"

        status_label, _ = self._get_status_from_score(score_percent)
        status = status_label.lower()

        return DimensionSummary(
            name="Privacy",
            score_display=score_display,
            score_percent=score_percent,
            description="Analyzes leakage risk via membership inference, exact matches, and other privacy tests.",
            highlights=highlights or ["Privacy metrics available"],
            status=status,
        )

    # ---------------------------------------------------------------- formatting
    @staticmethod
    def _get_status_from_score(score_percent: float) -> tuple[str, str]:
        """
        Get status label and class based on score percentage.
        
        Returns:
            tuple: (status_label, status_class)
            - status_label: "EXCELLENT", "GREAT", "GOOD", "FAIR", "POOR", or "SKIPPED"
            - status_class: CSS class name
        """
        if score_percent >= 90:
            return ("EXCELLENT", "status-excellent")
        elif score_percent >= 80:
            return ("GREAT", "status-great")
        elif score_percent >= 60:
            return ("GOOD", "status-good")
        elif score_percent >= 40:
            return ("FAIR", "status-fair")
        else:
            return ("POOR", "status-poor")

    @staticmethod
    def _normalize_score(value: Optional[float]) -> float:
        if value is None or not isinstance(value, (int, float)):
            return 0.0
        if math.isnan(value):
            return 0.0
        if value <= 1.0:
            return max(0.0, min(100.0, value * 100))
        return max(0.0, min(100.0, value))

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            if math.isnan(value):
                return None
            return float(value)
        if isinstance(value, str):
            try:
                parsed = float(value)
                if math.isnan(parsed):
                    return None
                return parsed
            except ValueError:
                return None
        return None

    @staticmethod
    def _format_percentage(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            base = value * 100 if abs(value) <= 1 else value
            return f"{base:.1f}%"
        return str(value)

    @staticmethod
    def _format_ratio(value: Any) -> Optional[str]:
        numeric = EvaluationHTMLGenerator._as_float(value)
        if numeric is None:
            return None
        return f"{numeric * 100:.1f}%"

    @staticmethod
    def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    # ------------------------------------------------------------------ rendering
    def _render_html(
        self,
        summaries: List[DimensionSummary],
        results: Dict[str, Any],
        command: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        command_block = f"<code>{command}</code>" if command else ""

        context_lines: List[str] = []
        if context:
            synthetic_path = context.get("synthetic_path")
            original_path = context.get("original_path")
            metadata_path = context.get("metadata_path")
            selected_dimensions = context.get("dimensions")

            if synthetic_path:
                context_lines.append(f"<strong>Synthetic:</strong> {synthetic_path}")
            if original_path:
                context_lines.append(f"<strong>Original:</strong> {original_path}")
            if metadata_path:
                context_lines.append(f"<strong>Metadata:</strong> {metadata_path}")
            if selected_dimensions:
                joined = ", ".join(selected_dimensions)
                context_lines.append(f"<strong>Dimensions:</strong> {joined}")

        context_html = "<br/>".join(context_lines)

        cards_html = "\n".join(self._render_card(summary) for summary in summaries)

        return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>SynEval Evaluation Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --glass-bg: rgba(255, 255, 255, 0.55);
      --glass-border: rgba(255, 255, 255, 0.38);
      --text-primary: rgba(16, 24, 32, 0.88);
      --text-secondary: rgba(60, 72, 88, 0.72);
      --accent-start: rgba(102, 126, 234, 0.9);
      --accent-end: rgba(118, 75, 162, 0.9);
      --frost-shadow: 0 24px 45px rgba(15, 23, 42, 0.28);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: radial-gradient(circle at top, #f8fbff 0%, #e7ebf8 45%, #d9deec 100%);
      color: var(--text-primary);
      margin: 0;
      padding: 0 0 64px;
      min-height: 100vh;
      position: relative;
      line-height: 1.6;
      letter-spacing: -0.01em;
      backdrop-filter: saturate(140%);
    }}

    body::before {{
      content: \"\";
      position: fixed;
      inset: 0;
      background: url(\"data:image/svg+xml,%3Csvg width='320' height='320' viewBox='0 0 320 320' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.12' fill-rule='evenodd'%3E%3Cpath d='M0 0h40v40H0zM80 80h40v40H80zM160 160h40v40h-40zM240 240h40v40h-40z'/%3E%3C/svg%3E\")
        repeat;
      opacity: 0.8;
      z-index: -2;
    }}

    header {{
      padding: 56px 12% 72px;
      background: linear-gradient(120deg, rgba(255, 255, 255, 0.65), rgba(255, 255, 255, 0.35));
      backdrop-filter: blur(26px) saturate(160%);
      border-bottom: 1px solid rgba(255, 255, 255, 0.4);
      box-shadow: 0 36px 70px rgba(90, 107, 143, 0.25);
      z-index: 10;
    }}

    header h1 {{
      margin: 0 0 16px;
      font-size: clamp(30px, 4vw, 40px);
      font-weight: 700;
      color: rgba(14, 20, 35, 0.92);
      letter-spacing: -0.015em;
    }}

    header p {{
      margin: 10px 0;
      font-size: 16px;
      color: var(--text-secondary);
    }}

    header code {{
      display: inline-block;
      overflow-wrap: anywhere;
      font-size: 13px;
      padding: 8px 12px;
      border-radius: 12px;
      background: rgba(28, 43, 72, 0.1);
      backdrop-filter: blur(18px);
      border: 1px solid rgba(255, 255, 255, 0.4);
    }}

    .container {{
      max-width: 1200px;
      margin: 32px auto 0;
      padding: 0 clamp(16px, 6vw, 64px);
    }}

    .context {{
      background: var(--glass-bg);
      border-radius: 22px;
      padding: 24px clamp(20px, 5vw, 42px);
      backdrop-filter: blur(22px);
      border: 1px solid var(--glass-border);
      box-shadow: 0 22px 45px rgba(15, 23, 42, 0.22);
      color: var(--text-secondary);
      margin-bottom: 42px;
    }}

    .context strong {{
      color: var(--text-primary);
    }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: clamp(24px, 4vw, 40px);
    }}

    .card {{
      background: var(--glass-bg);
      border-radius: 26px;
      padding: clamp(24px, 4vw, 32px);
      border: 1px solid var(--glass-border);
      box-shadow: var(--frost-shadow);
      backdrop-filter: blur(24px);
      transition: transform 0.4s ease, box-shadow 0.4s ease, border-color 0.4s ease;
      display: flex;
      flex-direction: column;
      align-items: flex-start;
    }}

    .card:hover {{
      transform: translateY(-10px);
      box-shadow: 0 32px 70px rgba(58, 78, 107, 0.35);
      border-color: rgba(255, 255, 255, 0.6);
    }}

    .card h2 {{
      margin: 0 0 20px;
      font-size: 20px;
      font-weight: 700;
      color: rgba(31, 41, 55, 0.85);
      letter-spacing: -0.01em;
    }}

    .score {{
      font-size: clamp(32px, 5vw, 42px);
      font-weight: 700;
      margin-bottom: 18px;
      letter-spacing: -0.03em;
    }}

    .score.ok {{ color: #2ecc71; }}
    .score.warning {{ color: #ed8936; }}
    .score.error {{ color: #e53e3e; }}
    .score.skipped {{ color: #718096; }}

    .progress-wrap {{
      background: rgba(255, 255, 255, 0.55);
      border-radius: 999px;
      height: 12px;
      margin-bottom: 22px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.4);
      width: 100%;
    }}

    .progress-bar {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--accent-start), var(--accent-end));
      transition: width 0.45s ease;
    }}

    .status-tag {{
      display: inline-block;
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      margin-bottom: 16px;
      background: rgba(255, 255, 255, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.4);
      color: rgba(30, 41, 59, 0.72);
      backdrop-filter: blur(18px);
    }}

    .status-excellent {{
      background: rgba(16, 185, 129, 0.22);
      color: #065f46;
      border-color: rgba(16, 185, 129, 0.3);
    }}

    .status-great {{
      background: rgba(72, 187, 120, 0.2);
      color: #18603e;
      border-color: rgba(72, 187, 120, 0.3);
    }}

    .status-good {{
      background: rgba(101, 163, 13, 0.18);
      color: #365314;
      border-color: rgba(101, 163, 13, 0.25);
    }}

    .status-fair {{
      background: rgba(237, 137, 54, 0.18);
      color: #9c4a14;
      border-color: rgba(237, 137, 54, 0.25);
    }}

    .status-poor {{
      background: rgba(229, 62, 62, 0.2);
      color: #8b1f1f;
      border-color: rgba(229, 62, 62, 0.3);
    }}

    .status-skipped {{
      background: rgba(160, 174, 192, 0.18);
      color: #364152;
      border-color: rgba(160, 174, 192, 0.25);
    }}

    ul {{
      padding-left: 20px;
      margin: 0;
      color: var(--text-secondary);
      font-size: 14px;
      line-height: 1.7;
      flex: 1;
    }}

    ul li {{
      margin-bottom: 8px;
    }}

    p {{
      margin: 0 0 18px;
      color: var(--text-secondary);
      font-size: 15px;
    }}

    footer {{
      margin-top: 72px;
      text-align: center;
      font-size: 13px;
      color: rgba(76, 86, 106, 0.72);
    }}

    @media (max-width: 1024px) {{
      .cards {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 600px) {{
      header {{
        padding: 36px clamp(24px, 8vw, 42px) 56px;
      }}

      header code {{
        width: 100%;
      }}

      .container {{
        margin-top: 32px;
      }}

      .card {{
        padding: 24px;
      }}

      .cards {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SynEval Evaluation Dashboard</h1>
    <p>Generated on {timestamp}</p>
    {f'<p>Command: {command_block}</p>' if command_block else ''}
  </header>
  <main class=\"container\">
    <section class=\"context\">
      {context_html if context_html else 'Evaluation summary for requested dimensions.'}
    </section>
    <section class=\"cards\">
      {cards_html}
    </section>
    <footer>
      SynEval &mdash; Synthetic Data Evaluation Framework
    </footer>
  </main>
</body>
</html>
"""

    def _render_card(self, summary: DimensionSummary) -> str:
        highlights_html = "".join(f"<li>{item}</li>" for item in summary.highlights)
        
        # Handle skipped status separately
        if summary.status == "skipped":
            status_label = "SKIPPED"
            status_class = "status-skipped"
            score_class = "skipped"
        else:
            # Get status from score
            status_label, status_class = self._get_status_from_score(summary.score_percent)
            # Map status to score color class
            score_class_map = {
                "excellent": "ok",
                "great": "ok",
                "good": "ok",
                "fair": "warning",
                "poor": "error",
            }
            score_class = score_class_map.get(summary.status, "ok")

        return f"""
      <article class=\"card\">
        <span class=\"status-tag {status_class}\">{status_label}</span>
        <h2>{summary.name}</h2>
        <div class=\"score {score_class}\">{summary.score_display}</div>
        <div class=\"progress-wrap\">
          <div class=\"progress-bar\" style=\"width: {summary.score_percent:.1f}%;\"></div>
        </div>
        <p>{summary.description}</p>
        <ul>{highlights_html}</ul>
      </article>
"""
