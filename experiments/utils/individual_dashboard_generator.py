#!/usr/bin/env python3
"""
Individual Experiment Dashboard Generator
=========================================
Generates beautiful, interactive HTML dashboards for individual DP experiments.
"""

import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from .privacy_scoring import compute_privacy_summary


class IndividualDashboardGenerator:
    """
    Generates stunning HTML dashboards for individual experiment results.
    """

    def __init__(self):
        """Initialize the dashboard generator."""
        self.primary_color = "#667eea"
        self.secondary_color = "#764ba2"
        self.success_color = "#28a745"
        self.warning_color = "#ffc107"
        self.danger_color = "#dc3545"
        self.info_color = "#17a2b8"

    def generate(self, results: Dict, experiment_name: Optional[str] = None) -> str:
        """
        Generate a complete HTML dashboard for an individual experiment.

        Args:
            results: Dictionary containing experiment results
            experiment_name: Optional display name for the experiment

        Returns:
            Complete HTML content as a string
        """
        exp_name = experiment_name or results.get(
            "group_display_name", results.get("experiment_id", "Unknown Experiment")
        )
        epsilon = results.get("epsilon", "Unknown")
        timestamp = results.get("timestamp", datetime.now().isoformat())

        # Generate all visualization components
        overview_cards = self._generate_overview_cards(results)
        fidelity_viz = self._generate_fidelity_section(results.get("fidelity", {}))
        utility_viz = self._generate_utility_section(results.get("utility", {}))
        privacy_viz = self._generate_privacy_section(results)
        diversity_viz = self._generate_diversity_section(results.get("diversity", {}))

        # Build the complete HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{exp_name} - ε={epsilon}</title>
    
    <!-- External Libraries -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        {self._generate_styles()}
    </style>
</head>
<body>
    <!-- Animated Background -->
    <div class="animated-bg"></div>
    
    <!-- Header -->
    <header class="dashboard-header">
        <div class="container-fluid">
            <div class="row align-items-center">
                <div class="col-lg-8">
                    <h1 class="display-5 mb-2">
                        <span class="dashboard-logo">
                            <img src="data:image/png;base64,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" 
                                alt="Secludy AI" 
                                style="height: 40px; display: block;" />
                        </span>
                        {exp_name}
                    </h1>
                    <div class="header-meta">
                        <span class="badge badge-epsilon">ε = {epsilon}</span>
                        <span class="text-light ms-3">
                            <i class="far fa-clock me-1"></i>
                            {self._format_timestamp(timestamp)}
                        </span>
                    </div>
                </div>
                <div class="col-lg-4 text-end">
                    <div class="header-stats">
                        <div class="stat-item">
                            <span class="stat-value" id="overall-score">--</span>
                            <span class="stat-label">Overall Score</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value" id="privacy-level">--</span>
                            <span class="stat-label">Privacy Level</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </header>
    
    <!-- Main Content -->
    <main class="container-fluid my-4">
        <!-- Overview Cards -->
        <section class="overview-section mb-5">
            {overview_cards}
        </section>
        
        <!-- Tabbed Content -->
        <div class="content-tabs">
            <ul class="nav nav-pills mb-4" id="metricTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="fidelity-tab" data-bs-toggle="tab" 
                            data-bs-target="#fidelity" type="button" role="tab">
                        <i class="fas fa-chart-line me-2"></i>
                        Fidelity
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="utility-tab" data-bs-toggle="tab" 
                            data-bs-target="#utility" type="button" role="tab">
                        <i class="fas fa-cogs me-2"></i>
                        Utility
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="privacy-tab" data-bs-toggle="tab" 
                            data-bs-target="#privacy" type="button" role="tab">
                        <i class="fas fa-lock me-2"></i>
                        Privacy
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="diversity-tab" data-bs-toggle="tab" 
                            data-bs-target="#diversity" type="button" role="tab">
                        <i class="fas fa-random me-2"></i>
                        Diversity
                    </button>
                </li>
            </ul>
            
            <div class="tab-content" id="metricTabsContent">
                <div class="tab-pane fade show active" id="fidelity" role="tabpanel">
                    {fidelity_viz}
                </div>
                <div class="tab-pane fade" id="utility" role="tabpanel">
                    {utility_viz}
                </div>
                <div class="tab-pane fade" id="privacy" role="tabpanel">
                    {privacy_viz}
                </div>
                <div class="tab-pane fade" id="diversity" role="tabpanel">
                    {diversity_viz}
                </div>
            </div>
        </div>
    </main>
    
    <!-- Footer -->
    <footer class="dashboard-footer">
        <div class="container-fluid">
            <div class="row">
                <div class="col-12 text-center">
                    <p class="mb-0">
                        <i class="fas fa-chart-bar me-2"></i>
                        Differential Privacy Evaluation Dashboard
                        <span class="mx-2">•</span>
                        Generated with SynEval
                    </p>
                </div>
            </div>
        </div>
    </footer>
    
    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        {self._generate_scripts(results)}
    </script>
</body>
</html>
"""
        return html

    def _generate_styles(self) -> str:
        """Generate the CSS styles for the dashboard."""
        return """
        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --success: #28a745;
            --warning: #ffc107;
            --danger: #dc3545;
            --info: #17a2b8;
            --dark: #2d3436;
            --light: #f8f9fa;
            --gradient-primary: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            --gradient-success: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            --gradient-warning: linear-gradient(135deg, #ffc107 0%, #ff8c00 100%);
            --gradient-danger: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            --shadow-sm: 0 2px 10px rgba(0,0,0,0.08);
            --shadow-md: 0 4px 20px rgba(0,0,0,0.12);
            --shadow-lg: 0 10px 40px rgba(0,0,0,0.15);
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }
        
        /* Logo styling */
        .dashboard-logo {
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 0.5rem 0.75rem;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.3);
            display: inline-flex;
            align-items: center;
            margin-right: 1.5rem;
        }

        .dashboard-logo img {
            height: 40px;
            display: block;
        }

        /* Animated Background */
        .animated-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            opacity: 0.05;
            background: 
                radial-gradient(circle at 20% 50%, var(--primary) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, var(--secondary) 0%, transparent 50%),
                radial-gradient(circle at 40% 20%, var(--info) 0%, transparent 50%);
            animation: bgAnimation 30s ease infinite;
        }
        
        @keyframes bgAnimation {
            0%, 100% { transform: translate(0, 0) scale(1); }
            25% { transform: translate(-20px, -20px) scale(1.05); }
            50% { transform: translate(20px, -10px) scale(1.1); }
            75% { transform: translate(-10px, 20px) scale(1.05); }
        }
        
        /* Header Styles */
        .dashboard-header {
            background: var(--gradient-primary);
            color: white;
            padding: 2.5rem 0;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }
        
        .dashboard-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 200%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .header-meta {
            margin-top: 1rem;
        }
        
        .badge-epsilon {
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        .header-stats {
            display: flex;
            gap: 2rem;
            justify-content: flex-end;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            display: block;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
        }
        
        .stat-label {
            display: block;
            font-size: 0.875rem;
            opacity: 0.9;
            margin-top: 0.25rem;
        }
        
        /* Overview Cards */
        .metric-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: var(--shadow-md);
            transition: var(--transition);
            position: relative;
            overflow: hidden;
            margin-bottom: 1.5rem;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow-lg);
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--gradient-primary);
        }
        
        .metric-card.success::before { background: var(--gradient-success); }
        .metric-card.warning::before { background: var(--gradient-warning); }
        .metric-card.danger::before { background: var(--gradient-danger); }
        
        .metric-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        }
        
        .metric-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--dark);
        }
        
        .metric-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--gradient-primary);
            color: white;
            font-size: 1.2rem;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--dark);
            line-height: 1;
        }
        
        .metric-subtitle {
            color: #6c757d;
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }
        
        .metric-change {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
            margin-top: 1rem;
        }
        
        .metric-change.positive {
            background: rgba(40, 167, 69, 0.1);
            color: var(--success);
        }
        
        .metric-change.negative {
            background: rgba(220, 53, 69, 0.1);
            color: var(--danger);
        }
        
        /* Navigation Tabs */
        .nav-pills {
            background: white;
            border-radius: 15px;
            padding: 0.5rem;
            box-shadow: var(--shadow-sm);
        }
        
        .nav-pills .nav-link {
            color: #6c757d;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: var(--transition);
            border: none;
        }
        
        .nav-pills .nav-link:hover {
            background: rgba(102, 126, 234, 0.1);
            color: var(--primary);
        }
        
        .nav-pills .nav-link.active {
            background: var(--gradient-primary);
            color: white;
            box-shadow: var(--shadow-md);
        }
        
        /* Content Sections */
        .content-section {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: var(--shadow-md);
            margin-bottom: 2rem;
        }
        
        .section-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--dark);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
        }
        
        .section-title i {
            margin-right: 0.75rem;
            color: var(--primary);
        }
        
        /* Charts Container */
        .chart-container {
            position: relative;
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
            margin-bottom: 1.5rem;
        }
        
        .chart-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--dark);
            margin-bottom: 1rem;
        }
        
        /* Gauge Charts */
        .gauge-container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 300px;
            height: 300px;
            width: 100%;
            max-width: 100%;
            overflow: hidden;
        }
        
        /* Progress Bars */
        .progress-item {
            margin-bottom: 1.5rem;
        }
        
        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: var(--dark);
        }
        
        .progress {
            height: 10px;
            border-radius: 5px;
            background: #e9ecef;
            overflow: visible;
        }
        
        .progress-bar {
            border-radius: 5px;
            position: relative;
            transition: width 1s ease-in-out;
        }
        
        .progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255,255,255,0.3);
            animation: progressShimmer 2s infinite;
        }
        
        @keyframes progressShimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        /* Status Badges */
        .status-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-badge.success {
            background: var(--gradient-success);
            color: white;
        }
        
        .status-badge.warning {
            background: var(--gradient-warning);
            color: white;
        }
        
        .status-badge.danger {
            background: var(--gradient-danger);
            color: white;
        }
        
        /* Alert Boxes */
        .custom-alert {
            border-radius: 10px;
            border: none;
            padding: 1rem 1.5rem;
            box-shadow: var(--shadow-sm);
        }
        
        .custom-alert.alert-success {
            background: linear-gradient(135deg, rgba(40, 167, 69, 0.1), rgba(32, 201, 151, 0.1));
            border-left: 4px solid var(--success);
            color: var(--success);
        }
        
        .custom-alert.alert-warning {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 140, 0, 0.1));
            border-left: 4px solid var(--warning);
            color: #856404;
        }
        
        .custom-alert.alert-danger {
            background: linear-gradient(135deg, rgba(220, 53, 69, 0.1), rgba(200, 35, 51, 0.1));
            border-left: 4px solid var(--danger);
            color: var(--danger);
        }
        
        .custom-alert.alert-info {
            background: linear-gradient(135deg, rgba(23, 162, 184, 0.1), rgba(13, 202, 240, 0.1));
            border-left: 4px solid var(--info);
            color: #0c5460;
        }
        
        /* Table Styles */
        .custom-table {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }
        
        .custom-table thead {
            background: var(--gradient-primary);
            color: white;
        }
        
        .custom-table th {
            border: none;
            padding: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.875rem;
            letter-spacing: 0.5px;
        }
        
        .custom-table td {
            padding: 1rem;
            border-top: 1px solid #e9ecef;
        }
        
        .custom-table tbody tr:hover {
            background: rgba(102, 126, 234, 0.05);
            transition: var(--transition);
        }
        
        /* Footer */
        .dashboard-footer {
            background: var(--dark);
            color: white;
            padding: 2rem 0;
            margin-top: 4rem;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .header-stats {
                justify-content: flex-start;
                margin-top: 1rem;
            }
            
            .nav-pills .nav-link {
                padding: 0.5rem 1rem;
                font-size: 0.875rem;
            }
            
            .metric-value {
                font-size: 2rem;
            }
        }
        
        /* Loading Animation */
        .loading-spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 2rem auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Tooltips */
        .tooltip-inner {
            background: var(--dark);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-size: 0.875rem;
        }
        """

    def _summarize_utility_metrics(self, tstr: Dict) -> Dict[str, Any]:
        """Derive task-aware utility metadata for dashboard rendering."""
        summary: Dict[str, Any] = {
            "available": False,
            "mode": "unknown",
            "score": 0.0,
            "metric_description": "ML task performance (TSTR)",
            "message": None,
            "task_type": tstr.get("task_type") if isinstance(tstr, dict) else None,
            "training_size": tstr.get("training_size")
            if isinstance(tstr, dict)
            else None,
            "test_size": tstr.get("test_size") if isinstance(tstr, dict) else None,
        }
        if not tstr or not isinstance(tstr, dict):
            return summary

        if tstr.get("error"):
            summary["message"] = tstr.get("error")
            return summary

        def normalize(report: Any) -> Dict[str, float]:
            """Coerce legacy classification reports into the metric layout used by the UI."""
            normalized: Dict[str, float] = {}
            if not isinstance(report, dict):
                return normalized

            for key in (
                "accuracy",
                "f1_macro",
                "precision_macro",
                "recall_macro",
                "rmse",
                "mae",
                "r2",
            ):
                value = report.get(key)
                if self._is_number(value):
                    normalized[key] = float(value)

            # Older utility payloads embed metrics inside classification_report-style blocks
            accuracy = report.get("accuracy")
            if self._is_number(accuracy):
                normalized.setdefault("accuracy", float(accuracy))

            macro_avg = report.get("macro avg")
            if isinstance(macro_avg, dict):
                mapping = {
                    "precision": "precision_macro",
                    "recall": "recall_macro",
                    "f1-score": "f1_macro",
                }
                for source, target in mapping.items():
                    value = macro_avg.get(source)
                    if self._is_number(value):
                        normalized.setdefault(target, float(value))

            weighted_avg = report.get("weighted avg")
            if isinstance(weighted_avg, dict):
                value = weighted_avg.get("f1-score")
                if self._is_number(value):
                    normalized.setdefault("f1_weighted", float(value))

            return normalized

        real_model_raw = tstr.get("real_data_model") or {}
        syn_model_raw = tstr.get("synthetic_data_model") or {}
        real_model = normalize(real_model_raw)
        syn_model = normalize(syn_model_raw)
        summary["real"] = real_model
        summary["synthetic"] = syn_model
        summary["raw_real_model"] = real_model_raw
        summary["raw_synthetic_model"] = syn_model_raw

        regression_keys = ["rmse", "mae", "r2"]
        classification_keys = ["accuracy", "f1_macro"]
        is_regression = any(self._is_number(syn_model.get(k)) for k in regression_keys)
        is_classification = any(
            self._is_number(syn_model.get(k)) for k in classification_keys
        )

        if is_regression:
            summary["mode"] = "regression"
            summary["available"] = True

            r2 = syn_model.get("r2")
            if self._is_number(r2):
                score_raw = (max(min(float(r2), 1.0), -1.0) + 1.0) * 50.0
                summary["score_metric_name"] = "R²"
                summary["score_metric_value"] = float(r2)
            else:
                score_raw = 0.0
                summary["score_metric_name"] = "R²"
                summary["score_metric_value"] = None
            summary["score"] = max(0.0, min(100.0, score_raw))
            summary["metric_description"] = "Synthetic R² (higher is better)"

            metrics = []
            real_values = []
            syn_values = []
            metric_map = [
                ("RMSE", real_model.get("rmse"), syn_model.get("rmse")),
                ("MAE", real_model.get("mae"), syn_model.get("mae")),
                ("R²", real_model.get("r2"), syn_model.get("r2")),
            ]
            for name, real_val, syn_val in metric_map:
                if self._is_number(real_val) or self._is_number(syn_val):
                    metrics.append(name)
                    real_values.append(
                        float(real_val) if self._is_number(real_val) else 0.0
                    )
                    syn_values.append(
                        float(syn_val) if self._is_number(syn_val) else 0.0
                    )

            summary["chart"] = {
                "metrics": metrics,
                "real_values": real_values,
                "synthetic_values": syn_values,
                "yaxis_title": "Metric Value",
                "rangemode": "tozero",
            }

        elif is_classification:
            summary["mode"] = "classification"
            summary["available"] = True

            accuracy = syn_model.get("accuracy")
            f1_macro = syn_model.get("f1_macro")

            if self._is_number(accuracy):
                score_raw = float(accuracy) * 100.0
                summary["score_metric_name"] = "Accuracy"
                summary["score_metric_value"] = float(accuracy)
                summary["metric_description"] = "Synthetic accuracy (higher is better)"
            elif self._is_number(f1_macro):
                score_raw = float(f1_macro) * 100.0
                summary["score_metric_name"] = "F1 Macro"
                summary["score_metric_value"] = float(f1_macro)
                summary["metric_description"] = "Synthetic F1 Macro (higher is better)"
            else:
                score_raw = 0.0
            summary["score"] = max(0.0, min(100.0, score_raw))

            metrics = []
            real_values = []
            syn_values = []
            if self._is_number(real_model.get("accuracy")) or self._is_number(
                syn_model.get("accuracy")
            ):
                metrics.append("Accuracy")
                real_values.append(
                    float(real_model.get("accuracy")) * 100.0
                    if self._is_number(real_model.get("accuracy"))
                    else 0.0
                )
                syn_values.append(
                    float(syn_model.get("accuracy")) * 100.0
                    if self._is_number(syn_model.get("accuracy"))
                    else 0.0
                )
            if self._is_number(real_model.get("f1_macro")) or self._is_number(
                syn_model.get("f1_macro")
            ):
                metrics.append("F1 Macro")
                real_values.append(
                    float(real_model.get("f1_macro")) * 100.0
                    if self._is_number(real_model.get("f1_macro"))
                    else 0.0
                )
                syn_values.append(
                    float(syn_model.get("f1_macro")) * 100.0
                    if self._is_number(syn_model.get("f1_macro"))
                    else 0.0
                )

            summary["chart"] = {
                "metrics": metrics,
                "real_values": real_values,
                "synthetic_values": syn_values,
                "yaxis_title": "Score (%)",
                "rangemode": "tozero",
                "yaxis_range": [0, 100],
            }
        else:
            summary["message"] = "Utility metrics unavailable in results payload."

        return summary

    def _summarize_privacy_metrics(self, results: Dict) -> Dict[str, Any]:
        """Wrap the shared privacy scoring helper."""
        return compute_privacy_summary(results or {})

    def _is_number(self, value: Any) -> bool:
        """Return True when value is a real number (excluding NaN)."""
        if isinstance(value, (int, float)):
            if isinstance(value, float) and math.isnan(value):
                return False
            return True
        return False

    def _format_float(self, value: Any, precision: int = 4) -> str:
        """Format numeric values with fixed precision, handling missing cases."""
        if not self._is_number(value):
            return "N/A"
        return f"{float(value):.{precision}f}"

    def _format_percent(self, value: Any, precision: int = 1) -> str:
        """Format fractional values as percentages."""
        if not self._is_number(value):
            return "N/A"
        return f"{float(value) * 100:.{precision}f}%"

    def _generate_overview_cards(self, results: Dict) -> str:
        """Generate overview metric cards."""
        cards_html = '<div class="row">'

        # Fidelity Score Card
        fidelity_score = (
            self._safe_get(results, "fidelity.quality.Overall.score", 0) * 100
        )
        fidelity_class = self._get_score_class(fidelity_score)
        cards_html += f"""
        <div class="col-lg-3 col-md-6">
            <div class="metric-card {fidelity_class}">
                <div class="metric-header">
                    <span class="metric-title">Fidelity Score</span>
                    <div class="metric-icon">
                        <i class="fas fa-chart-line"></i>
                    </div>
                </div>
                <div class="metric-value">{fidelity_score:.1f}%</div>
                <div class="metric-subtitle">Statistical similarity to real data</div>
                <div class="metric-change {self._get_change_class(fidelity_score)}">
                    <i class="fas fa-arrow-{self._get_arrow_direction(fidelity_score)} me-1"></i>
                    {self._get_score_label(fidelity_score)}
                </div>
            </div>
        </div>
        """

        # Utility Score Card
        utility_summary = self._summarize_utility_metrics(
            results.get("utility", {}).get("tstr_accuracy", {})
        )
        utility_score = utility_summary.get("score", 0.0)
        utility_class = self._get_score_class(utility_score)
        utility_subtitle = utility_summary.get(
            "metric_description", "ML task performance (TSTR)"
        )
        cards_html += f"""
        <div class="col-lg-3 col-md-6">
            <div class="metric-card {utility_class}">
                <div class="metric-header">
                    <span class="metric-title">Utility Score</span>
                    <div class="metric-icon">
                        <i class="fas fa-cogs"></i>
                    </div>
                </div>
                <div class="metric-value">{utility_score:.1f}%</div>
                <div class="metric-subtitle">{utility_subtitle}</div>
                <div class="metric-change {self._get_change_class(utility_score)}">
                    <i class="fas fa-arrow-{self._get_arrow_direction(utility_score)} me-1"></i>
                    {self._get_score_label(utility_score)}
                </div>
            </div>
        </div>
        """

        # Privacy Score Card (inverted - lower is better)
        privacy_summary = self._summarize_privacy_metrics(results)
        privacy_score_value = privacy_summary.get("score")
        if privacy_score_value is None:
            privacy_score_display = "N/A"
            privacy_class = "warning"
            change_html = """
                <div class="metric-change text-muted">
                    <i class="fas fa-question me-1"></i>
                    No Data
                </div>
            """
        else:
            privacy_class = self._get_score_class(privacy_score_value)
            change_class = self._get_change_class(privacy_score_value)
            score_label = self._get_score_label(privacy_score_value)
            privacy_score_display = f"{privacy_score_value:.1f}%"
            change_html = f"""
                <div class="metric-change {change_class}">
                    <i class="fas fa-arrow-{self._get_arrow_direction(privacy_score_value)} me-1"></i>
                    {score_label}
                </div>
            """
        if privacy_summary.get("notes"):
            note_hint = privacy_summary["notes"][0]
            privacy_subtitle = (
                note_hint
                if privacy_score_value is not None and privacy_score_value < 80
                else privacy_summary.get("subtitle", "Structured privacy metrics")
            )
        else:
            privacy_subtitle = privacy_summary.get(
                "subtitle", "Structured privacy metrics"
            )
        cards_html += f"""
        <div class="col-lg-3 col-md-6">
            <div class="metric-card {privacy_class}">
                <div class="metric-header">
                    <span class="metric-title">Privacy Score</span>
                    <div class="metric-icon">
                        <i class="fas fa-lock"></i>
                    </div>
                </div>
                <div class="metric-value">{privacy_score_display}</div>
                <div class="metric-subtitle">{privacy_subtitle}</div>
                {change_html}
            </div>
        </div>
        """

        # Diversity Score Card
        entropy_ratio = self._safe_get(
            results,
            "diversity.tabular_diversity.entropy_metrics.dataset_entropy.entropy_ratio",
            0,
        )
        diversity_score = entropy_ratio * 100
        diversity_class = self._get_score_class(diversity_score)
        cards_html += f"""
        <div class="col-lg-3 col-md-6">
            <div class="metric-card {diversity_class}">
                <div class="metric-header">
                    <span class="metric-title">Diversity Score</span>
                    <div class="metric-icon">
                        <i class="fas fa-random"></i>
                    </div>
                </div>
                <div class="metric-value">{diversity_score:.1f}%</div>
                <div class="metric-subtitle">Data variety (entropy ratio)</div>
                <div class="metric-change {self._get_change_class(diversity_score)}">
                    <i class="fas fa-arrow-{self._get_arrow_direction(diversity_score)} me-1"></i>
                    {self._get_score_label(diversity_score)}
                </div>
            </div>
        </div>
        """

        cards_html += "</div>"
        return cards_html

    def _generate_fidelity_section(self, fidelity_data: Dict) -> str:
        """Generate the fidelity visualization section."""
        if not fidelity_data:
            return self._empty_section("Fidelity")

        html = '<div class="content-section">'
        html += '<h2 class="section-title"><i class="fas fa-chart-line"></i>Fidelity Analysis</h2>'

        # Diagnostic and Quality Scores
        html += '<div class="row mb-4">'

        # Diagnostic Score Gauge
        diagnostic_score = (
            fidelity_data.get("diagnostic", {}).get("Overall", {}).get("score", 0) * 100
        )
        html += f"""
        <div class="col-md-6">
            <div class="chart-container">
                <h4 class="chart-title">Diagnostic Score</h4>
                <div id="diagnostic-gauge" class="gauge-container"></div>
                <div class="text-center mt-2">
                    <span class="status-badge {self._get_score_class(diagnostic_score)}">
                        {self._get_score_label(diagnostic_score)}
                    </span>
                </div>
            </div>
        </div>
        """

        # Quality Score Gauge
        quality_score = (
            fidelity_data.get("quality", {}).get("Overall", {}).get("score", 0) * 100
        )
        html += f"""
        <div class="col-md-6">
            <div class="chart-container">
                <h4 class="chart-title">Quality Score</h4>
                <div id="quality-gauge" class="gauge-container"></div>
                <div class="text-center mt-2">
                    <span class="status-badge {self._get_score_class(quality_score)}">
                        {self._get_score_label(quality_score)}
                    </span>
                </div>
            </div>
        </div>
        """
        html += "</div>"

        # Numerical Statistics Comparison
        num_stats = fidelity_data.get("numerical_statistics", {})
        if num_stats:
            html += '<div class="chart-container">'
            html += '<h4 class="chart-title">Column Statistics Comparison</h4>'
            html += '<div id="stats-comparison-chart" style="height: 400px;"></div>'
            html += "</div>"

            # Statistics Table
            html += self._generate_stats_table(num_stats)

        # Distinguishability AUC
        dist_auc = fidelity_data.get("distinguishability_auc", 0)
        html += f"""
        <div class="custom-alert alert-{'danger' if dist_auc > 0.8 else 'warning' if dist_auc > 0.6 else 'success'}">
            <strong>Distinguishability Test:</strong> AUC = {dist_auc:.4f}
            <br>
            <small>
                {self._interpret_distinguishability(dist_auc)}
            </small>
        </div>
        """

        html += "</div>"

        # Generate Plotly scripts for gauges and charts
        html += self._generate_fidelity_scripts(fidelity_data)

        return html

    def _generate_utility_section(self, utility_data: Dict) -> str:
        """Generate the utility visualization section."""
        if not utility_data:
            return self._empty_section("Utility")

        tstr = utility_data.get("tstr_accuracy", {})
        summary = self._summarize_utility_metrics(tstr)
        if not summary.get("available"):
            return self._empty_section("Utility", summary.get("message"))

        real_model = summary.get("real", {})
        syn_model = summary.get("synthetic", {})

        html = '<div class="content-section">'
        html += (
            '<h2 class="section-title"><i class="fas fa-cogs"></i>Utility Analysis</h2>'
        )
        html += '<div class="row">'
        html += """
        <div class="col-md-8">
            <div class="chart-container">
                <h4 class="chart-title">TSTR Performance Comparison</h4>
                <div id="tstr-comparison-chart" style="height: 400px;"></div>
            </div>
        </div>
        """

        training_size = summary.get("training_size")
        test_size = summary.get("test_size")

        def _format_count(value: Any) -> str:
            return f"{int(value):,}" if isinstance(value, int) and value >= 0 else "N/A"

        if summary.get("mode") == "classification":

            def bar_class(val: Any) -> str:
                if not self._is_number(val):
                    return "bg-secondary"
                val = float(val)
                if val >= 0.8:
                    return "bg-success"
                if val >= 0.6:
                    return "bg-warning"
                return "bg-danger"

            def percent_width(val: Any) -> str:
                if not self._is_number(val):
                    return "0%"
                pct = max(0.0, min(float(val) * 100.0, 100.0))
                return f"{pct:.1f}%"

            def classification_block(label: str, real_val: Any, syn_val: Any) -> str:
                if not (self._is_number(real_val) or self._is_number(syn_val)):
                    return ""
                return f"""
                <div class="progress-item">
                    <div class="progress-label">
                        <span>{label} (Real)</span>
                        <span>{self._format_percent(real_val, 1)}</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar {bar_class(real_val)}" style="width: {percent_width(real_val)}"></div>
                    </div>
                </div>
                <div class="progress-item">
                    <div class="progress-label">
                        <span>{label} (Synthetic)</span>
                        <span>{self._format_percent(syn_val, 1)}</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar {bar_class(syn_val)}" style="width: {percent_width(syn_val)}"></div>
                    </div>
                </div>
                """

            accuracy_block = classification_block(
                "Accuracy", real_model.get("accuracy"), syn_model.get("accuracy")
            )
            f1_block = classification_block(
                "F1 Macro", real_model.get("f1_macro"), syn_model.get("f1_macro")
            )

            html += """
            <div class="col-md-4">
                <div class="chart-container">
                    <h4 class="chart-title">Key Metrics</h4>
            """
            html += accuracy_block
            html += f1_block
            html += f"""
                    <hr>
                    <p class="text-muted small">
                        <strong>Task:</strong> {summary.get('task_type', 'Unknown')}<br>
                        <strong>Training Size:</strong> {_format_count(training_size)}<br>
                        <strong>Test Size:</strong> {_format_count(test_size)}
                    </p>
                </div>
            </div>
            """
        else:  # regression or unknown numeric metrics
            metric_rows = []
            for label, key in [("RMSE", "rmse"), ("MAE", "mae"), ("R²", "r2")]:
                real_val = real_model.get(key)
                syn_val = syn_model.get(key)
                if self._is_number(real_val) or self._is_number(syn_val):
                    if self._is_number(syn_val) and self._is_number(real_val):
                        diff = float(syn_val) - float(real_val)
                        diff_display = f"{diff:+.4f}"
                    else:
                        diff_display = "N/A"
                    metric_rows.append(
                        f"<tr><td><strong>{label}</strong></td>"
                        f"<td>{self._format_float(real_val)}</td>"
                        f"<td>{self._format_float(syn_val)}</td>"
                        f"<td>{diff_display}</td></tr>"
                    )

            metrics_table = ""
            if metric_rows:
                metrics_table = """
                <table class="table custom-table table-sm">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Real</th>
                            <th>Synthetic</th>
                            <th>Δ (Syn - Real)</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                metrics_table += "".join(metric_rows)
                metrics_table += """
                    </tbody>
                </table>
                """

            html += """
            <div class="col-md-4">
                <div class="chart-container">
                    <h4 class="chart-title">Key Metrics</h4>
            """
            html += (
                metrics_table
                or '<p class="text-muted small">No numeric metrics available.</p>'
            )
            html += f"""
                    <hr>
                    <p class="text-muted small">
                        <strong>Task:</strong> {summary.get('task_type', 'Unknown')}<br>
                        <strong>Training Size:</strong> {_format_count(training_size)}<br>
                        <strong>Test Size:</strong> {_format_count(test_size)}
                    </p>
                </div>
            </div>
            """

        html += "</div>"  # close row

        # Input/Output Columns Info
        input_columns = tstr.get("input_columns", []) or []
        output_columns = tstr.get("output_columns", []) or []
        html += f"""
        <div class="custom-alert alert-info mt-3">
            <strong>Model Configuration:</strong><br>
            Input Columns: {', '.join(input_columns) if input_columns else 'N/A'}<br>
            Output Columns: {', '.join(output_columns) if output_columns else 'N/A'}
        </div>
        """

        html += "</div>"
        html += self._generate_utility_scripts(utility_data)

        return html

    def _generate_privacy_section(self, results: Dict) -> str:
        """Generate the privacy visualization section."""
        summary = self._summarize_privacy_metrics(results)
        if not summary.get("available"):
            return self._empty_section("Privacy", summary.get("message"))

        html = '<div class="content-section">'
        html += (
            '<h2 class="section-title"><i class="fas fa-lock"></i>Privacy Analysis</h2>'
        )

        overall_score = summary.get("score") or 0.0
        level = summary.get("level", "Unknown")
        color = summary.get("color", "#6c757d")
        message = summary.get("message")

        html += f"""
        <div class="metric-card" style="border-left: 6px solid {color};">
            <div class="metric-header">
                <span class="metric-title">Overall Privacy Score</span>
                <div class="metric-icon" style="background: {color};">
                    <i class="fas fa-lock"></i>
                </div>
            </div>
            <div class="metric-value">{overall_score:.1f}%</div>
            <div class="metric-subtitle">{summary.get('subtitle', 'Structured privacy metrics')}</div>
            <span class="status-badge {'success' if overall_score >= 80 else 'warning' if overall_score >= 60 else 'danger'}">{level}</span>
            {f'<p class="text-muted small mt-2 mb-0">{message}</p>' if message else ''}
        </div>
        """

        components = summary.get("components", [])
        if components:
            html += '<div class="row g-3 mt-1">'
            for component in components:
                comp_score = component.get("score")
                comp_class = (
                    self._get_score_class(comp_score)
                    if comp_score is not None
                    else "warning"
                )
                score_display = (
                    f"{comp_score:.1f}%" if comp_score is not None else "N/A"
                )
                value = component.get("value")
                value_display = (
                    self._format_float(value, 4) if value is not None else "N/A"
                )
                extras = []
                if component.get("baseline") is not None:
                    extras.append(
                        f"Baseline: {self._format_float(component['baseline'], 4)}"
                    )
                if component.get("target") is not None:
                    extras.append(
                        f"Target: {self._format_float(component['target'], 4)}"
                    )
                direction_hint = (
                    "Higher values indicate stronger privacy."
                    if component.get("direction") == "higher_better"
                    else "Lower values indicate stronger privacy."
                )
                description = component.get("description", "")
                html += f"""
                <div class="col-md-4">
                    <div class="metric-card {comp_class}">
                        <div class="metric-header">
                            <span class="metric-title">{component.get('label', component.get('name', 'Metric'))}</span>
                            <div class="metric-icon">
                                <i class="fas fa-shield-alt"></i>
                            </div>
                        </div>
                        <div class="metric-value">{score_display}</div>
                        <div class="metric-subtitle">Current value: {value_display}</div>
                        <p class="text-muted small mb-1">{direction_hint}</p>
                        {f'<p class="text-muted small mb-0">{description}</p>' if description else ''}
                        {''.join(f'<p class="text-muted small mb-0">{line}</p>' for line in extras)}
                    </div>
                </div>
                """
            html += "</div>"

        if summary.get("notes"):
            note_items = "".join(f"<li>{note}</li>" for note in summary["notes"])
            html += f"""
            <div class="custom-alert alert-warning mt-3">
                <strong>Privacy Watchouts:</strong>
                <ul class="mb-0">{note_items}</ul>
            </div>
            """

        structured_metrics = summary.get("structured_metrics") or {}
        if structured_metrics:
            html += '<div class="table-responsive mt-3">'
            html += '<table class="table table-sm table-striped align-middle">'
            html += "<thead><tr><th>Metric</th><th>Synthetic</th><th>Baseline</th><th>Details</th></tr></thead><tbody>"
            for metric_name, metric in structured_metrics.items():
                if not isinstance(metric, dict):
                    continue
                syn_value = metric.get("syn_train_5pct") or metric.get("ims_syn_train")
                base_value = metric.get("train_train_5pct")
                desc = (
                    metric.get("desc")
                    or metric.get("method")
                    or "No description provided."
                )
                html += f"<tr><td>{metric_name}</td><td>{self._format_float(syn_value, 4)}</td><td>{self._format_float(base_value, 4)}</td><td class='small text-muted'>{desc}</td></tr>"
            html += "</tbody></table></div>"

        anonymeter = summary.get("anonymeter")
        if isinstance(anonymeter, dict) and anonymeter:
            html += '<div class="table-responsive mt-3">'
            html += '<table class="table table-sm">'
            html += "<thead><tr><th>Anonymeter Metric</th><th>Risk</th><th>Description</th></tr></thead><tbody>"
            for name, block in anonymeter.items():
                if not isinstance(block, dict):
                    continue
                risk = block.get("risk")
                desc = (
                    block.get("desc")
                    or block.get("description")
                    or "No description provided."
                )
                html += f"<tr><td>{name}</td><td>{self._format_float(risk, 4)}</td><td class='small text-muted'>{desc}</td></tr>"
            html += "</tbody></table></div>"

        html += "</div>"
        return html

    def _generate_diversity_section(self, diversity_data: Dict) -> str:
        """Generate the diversity visualization section."""
        if not diversity_data:
            return self._empty_section("Diversity")

        html = '<div class="content-section">'
        html += '<h2 class="section-title"><i class="fas fa-random"></i>Diversity Analysis</h2>'

        tabular = diversity_data.get("tabular_diversity", {})
        if tabular:
            # Coverage Chart
            coverage = tabular.get("coverage", {})
            if coverage:
                html += """
                <div class="chart-container mb-4">
                    <h4 class="chart-title">Column Coverage</h4>
                    <div id="coverage-chart" style="height: 300px;"></div>
                </div>
                """

            # Uniqueness Metrics
            uniqueness = tabular.get("uniqueness", {})
            if uniqueness:
                syn_dup = uniqueness.get("synthetic_duplicate_ratio", 0)
                orig_dup = uniqueness.get("original_duplicate_ratio", 0)
                rel_dup = uniqueness.get("relative_duplication", 0)

                html += f"""
                <div class="row mb-4">
                    <div class="col-md-4">
                        <div class="chart-container text-center">
                            <h5>Synthetic Duplicates</h5>
                            <div class="display-4 text-primary">{syn_dup:.1f}%</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="chart-container text-center">
                            <h5>Original Duplicates</h5>
                            <div class="display-4 text-secondary">{orig_dup:.1f}%</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="chart-container text-center">
                            <h5>Relative Duplication</h5>
                            <div class="display-4 text-{'success' if rel_dup < 50 else 'warning'}">{rel_dup:.1f}%</div>
                        </div>
                    </div>
                </div>
                """

            # Entropy Comparison
            entropy = tabular.get("entropy_metrics", {}).get("dataset_entropy", {})
            if entropy:
                html += """
                <div class="chart-container">
                    <h4 class="chart-title">Entropy Comparison</h4>
                    <div id="entropy-chart" style="height: 300px;"></div>
                </div>
                """

        html += "</div>"
        html += self._generate_diversity_scripts(diversity_data)

        return html

    def _generate_stats_table(self, num_stats: Dict) -> str:
        """Generate a statistics comparison table."""
        html = """
        <div class="table-responsive mt-4">
            <table class="table custom-table">
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Mean Diff</th>
                        <th>Std Diff</th>
                        <th>KL Divergence</th>
                        <th>Fidelity Score</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """

        for col_name, col_data in num_stats.items():
            if isinstance(col_data, dict) and "overall_fidelity_score" in col_data:
                rel_diff = col_data.get("relative_differences", {})
                dist_sim = col_data.get("distribution_similarity", {})
                fidelity_score = col_data.get("overall_fidelity_score", 0)

                status_class = self._get_score_class(fidelity_score * 100)

                html += f"""
                    <tr>
                        <td><strong>{col_name}</strong></td>
                        <td>{rel_diff.get('mean_diff', 0):.2f}x</td>
                        <td>{rel_diff.get('std_diff', 0):.2f}x</td>
                        <td>{dist_sim.get('kl_divergence', 0):.3f}</td>
                        <td>{fidelity_score:.3f}</td>
                        <td>
                            <span class="badge bg-{status_class}">
                                {self._get_score_label(fidelity_score * 100)}
                            </span>
                        </td>
                    </tr>
                """

        html += """
                </tbody>
            </table>
        </div>
        """
        return html

    def _generate_fidelity_scripts(self, fidelity_data: Dict) -> str:
        """Generate JavaScript for fidelity visualizations."""
        diagnostic_score = (
            fidelity_data.get("diagnostic", {}).get("Overall", {}).get("score", 0) * 100
        )
        quality_score = (
            fidelity_data.get("quality", {}).get("Overall", {}).get("score", 0) * 100
        )

        # Prepare data for statistics comparison chart
        num_stats = fidelity_data.get("numerical_statistics", {})
        columns = []
        mean_diffs = []
        std_diffs = []

        for col_name, col_data in num_stats.items():
            if isinstance(col_data, dict) and "relative_differences" in col_data:
                columns.append(col_name)
                rel_diff = col_data["relative_differences"]
                mean_diffs.append(rel_diff.get("mean_diff", 0))
                std_diffs.append(rel_diff.get("std_diff", 0))

        return f"""
        <script>
        (function() {{
            // Diagnostic Gauge
            var diagnosticData = [{{
                type: 'indicator',
                mode: 'gauge+number',
                value: {diagnostic_score:.1f},
                title: {{text: 'Diagnostic Score', font: {{size: 16}}}},
                gauge: {{
                    axis: {{range: [0, 100]}},
                    bar: {{color: '{self._get_gauge_color(diagnostic_score)}'}},
                    bgcolor: 'white',
                    borderwidth: 2,
                    bordercolor: 'gray',
                    steps: [
                        {{range: [0, 40], color: 'rgba(220, 53, 69, 0.1)'}},
                        {{range: [40, 70], color: 'rgba(255, 193, 7, 0.1)'}},
                        {{range: [70, 100], color: 'rgba(40, 167, 69, 0.1)'}}
                    ]
                }}
            }}];
            
            var gaugeLayout = {{
                width: 400,
                height: 300,
                margin: {{t: 50, r: 50, l: 50, b: 25}},
                paper_bgcolor: 'transparent',
                font: {{family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto'}}
            }};
            
            Plotly.newPlot('diagnostic-gauge', diagnosticData, gaugeLayout, {{responsive: true, displayModeBar: false}});
            
            // Quality Gauge
            var qualityData = [{{
                type: 'indicator',
                mode: 'gauge+number',
                value: {quality_score:.1f},
                title: {{text: 'Quality Score', font: {{size: 16}}}},
                gauge: {{
                    axis: {{range: [0, 100]}},
                    bar: {{color: '{self._get_gauge_color(quality_score)}'}},
                    bgcolor: 'white',
                    borderwidth: 2,
                    bordercolor: 'gray',
                    steps: [
                        {{range: [0, 40], color: 'rgba(220, 53, 69, 0.1)'}},
                        {{range: [40, 70], color: 'rgba(255, 193, 7, 0.1)'}},
                        {{range: [70, 100], color: 'rgba(40, 167, 69, 0.1)'}}
                    ]
                }}
            }}];
            
            Plotly.newPlot('quality-gauge', qualityData, gaugeLayout, {{responsive: true, displayModeBar: false}});
            
            // Statistics Comparison Chart
            if (document.getElementById('stats-comparison-chart')) {{
                var trace1 = {{
                    x: {columns},
                    y: {mean_diffs},
                    name: 'Mean Difference',
                    type: 'bar',
                    marker: {{color: 'rgba(102, 126, 234, 0.8)'}}
                }};
                
                var trace2 = {{
                    x: {columns},
                    y: {std_diffs},
                    name: 'Std Difference',
                    type: 'bar',
                    marker: {{color: 'rgba(118, 75, 162, 0.8)'}}
                }};
                
                var layout = {{
                    title: 'Relative Differences (Synthetic vs Real)',
                    barmode: 'group',
                    yaxis: {{
                        title: 'Relative Difference (x times)',
                        zeroline: true,
                        zerolinewidth: 2,
                        zerolinecolor: '#969696'
                    }},
                    xaxis: {{title: 'Columns'}},
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'rgba(0,0,0,0.02)',
                    font: {{family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto'}}
                }};
                
                Plotly.newPlot('stats-comparison-chart', [trace1, trace2], layout, {{responsive: true}});
            }}
        }})();
        </script>
        """

    def _generate_utility_scripts(self, utility_data: Dict) -> str:
        """Generate JavaScript for utility visualizations."""
        tstr = utility_data.get("tstr_accuracy", {})
        summary = self._summarize_utility_metrics(tstr)
        chart = summary.get("chart", {}) if isinstance(summary, dict) else {}
        metrics = chart.get("metrics", [])
        if not summary.get("available") or not metrics:
            return ""

        metrics_json = json.dumps(metrics)
        real_json = json.dumps(chart.get("real_values", []))
        syn_json = json.dumps(chart.get("synthetic_values", []))
        yaxis_config = {
            "title": chart.get("yaxis_title", "Metric Value"),
            "rangemode": chart.get("rangemode", "tozero"),
        }
        if chart.get("yaxis_range"):
            yaxis_config["range"] = chart.get("yaxis_range")
        yaxis_json = json.dumps(yaxis_config)

        return f"""
        <script>
        (function() {{
            var metrics = {metrics_json};
            if (!metrics.length) {{
                return;
            }}
            var realValues = {real_json};
            var synValues = {syn_json};
            
            var trace1 = {{
                x: metrics,
                y: realValues,
                name: 'Real Data Model',
                type: 'bar',
                marker: {{
                    color: 'rgba(40, 167, 69, 0.8)',
                    line: {{width: 2, color: 'rgba(40, 167, 69, 1)'}}
                }}
            }};
            
            var trace2 = {{
                x: metrics,
                y: synValues,
                name: 'Synthetic Data Model',
                type: 'bar',
                marker: {{
                    color: 'rgba(102, 126, 234, 0.8)',
                    line: {{width: 2, color: 'rgba(102, 126, 234, 1)'}}
                }}
            }};
            
            var layout = {{
                title: 'Train on Synthetic, Test on Real (TSTR)',
                barmode: 'group',
                yaxis: {yaxis_json},
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'rgba(0,0,0,0.02)',
                font: {{family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto'}},
                hovermode: 'x unified'
            }};
            
            Plotly.newPlot('tstr-comparison-chart', [trace1, trace2], layout, {{responsive: true}});
        }})();
        </script>
        """

    def _generate_diversity_scripts(self, diversity_data: Dict) -> str:
        """Generate JavaScript for diversity visualizations."""
        tabular = diversity_data.get("tabular_diversity", {})
        coverage = tabular.get("coverage", {})
        entropy_data = tabular.get("entropy_metrics", {}).get("dataset_entropy", {})

        # Prepare coverage data
        columns = list(coverage.keys())
        coverage_values = list(coverage.values())

        return f"""
        <script>
        (function() {{
            // Coverage Chart
            if (document.getElementById('coverage-chart')) {{
                var trace = {{
                    x: {columns},
                    y: {coverage_values},
                    type: 'bar',
                    marker: {{
                        color: {coverage_values},
                        colorscale: [
                            [0, 'rgba(220, 53, 69, 0.8)'],
                            [0.5, 'rgba(255, 193, 7, 0.8)'],
                            [1, 'rgba(40, 167, 69, 0.8)']
                        ],
                        cmin: 0,
                        cmax: 100,
                        showscale: true,
                        colorbar: {{
                            title: 'Coverage %',
                            thickness: 15
                        }}
                    }}
                }};
                
                var layout = {{
                    title: 'Column Coverage Percentage',
                    yaxis: {{
                        title: 'Coverage (%)',
                        range: [0, 105]
                    }},
                    xaxis: {{title: 'Columns'}},
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'rgba(0,0,0,0.02)',
                    font: {{family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto'}}
                }};
                
                Plotly.newPlot('coverage-chart', [trace], layout, {{responsive: true}});
            }}
            
            // Entropy Chart
            if (document.getElementById('entropy-chart')) {{
                var data = [{{
                    labels: ['Real Data', 'Synthetic Data'],
                    values: [
                        {entropy_data.get('real', 0):.2f},
                        {entropy_data.get('synthetic', 0):.2f}
                    ],
                    type: 'pie',
                    hole: 0.4,
                    marker: {{
                        colors: ['rgba(40, 167, 69, 0.8)', 'rgba(102, 126, 234, 0.8)']
                    }},
                    textinfo: 'label+value',
                    hovertemplate: '%{{label}}: %{{value:.2f}}<extra></extra>'
                }}];
                
                var layout = {{
                    title: 'Dataset Entropy Comparison',
                    annotations: [{{
                        font: {{size: 20}},
                        showarrow: false,
                        text: 'Entropy',
                        x: 0.5,
                        y: 0.5
                    }}],
                    paper_bgcolor: 'transparent',
                    font: {{family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto'}}
                }};
                
                Plotly.newPlot('entropy-chart', data, layout, {{responsive: true}});
            }}
        }})();
        </script>
        """

    def _generate_scripts(self, results: Dict) -> str:
        """Generate general dashboard scripts."""
        # Calculate overall score
        fidelity_score = (
            self._safe_get(results, "fidelity.quality.Overall.score", 0) * 100
        )
        utility_summary = self._summarize_utility_metrics(
            results.get("utility", {}).get("tstr_accuracy", {})
        )
        utility_score = utility_summary.get("score", 0.0)
        privacy_summary = self._summarize_privacy_metrics(results)
        privacy_score = privacy_summary.get("score")
        diversity_ratio = self._safe_get(
            results,
            "diversity.tabular_diversity.entropy_metrics.dataset_entropy.entropy_ratio",
            0,
        )
        diversity_score = diversity_ratio * 100

        components = [fidelity_score, utility_score, diversity_score]
        if privacy_score is not None:
            components.append(privacy_score)
        overall_score = sum(components) / len(components) if components else 0

        # Determine privacy level
        privacy_level = privacy_summary.get("level", "Low")
        privacy_color = privacy_summary.get("color", "#dc3545")
        reliability = privacy_summary.get("reliability")
        if reliability == "low":
            privacy_level_display = f"{privacy_level} (attack unreliable)"
        elif reliability == "medium":
            privacy_level_display = f"{privacy_level} (moderate reliability)"
        else:
            privacy_level_display = privacy_level

        return f"""
        document.addEventListener('DOMContentLoaded', function() {{
            // Update header statistics
            document.getElementById('overall-score').textContent = '{overall_score:.1f}%';
            document.getElementById('privacy-level').textContent = '{privacy_level_display}';
            document.getElementById('privacy-level').style.color = '{privacy_color}';
            
            // Add smooth scroll behavior
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                anchor.addEventListener('click', function (e) {{
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {{
                        target.scrollIntoView({{behavior: 'smooth'}});
                    }}
                }});
            }});
            
            // Initialize tooltips
            var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
            var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {{
                return new bootstrap.Tooltip(tooltipTriggerEl)
            }});
            
            // Animate numbers on scroll
            const observerOptions = {{
                threshold: 0.1,
                rootMargin: '0px'
            }};
            
            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        entry.target.classList.add('animate-in');
                    }}
                }});
            }}, observerOptions);
            
            document.querySelectorAll('.metric-card').forEach(card => {{
                observer.observe(card);
            }});
        }});
        """

    # Helper methods
    def _safe_get(self, data: Dict, path: str, default: Any = None) -> Any:
        """Safely get nested dictionary values."""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value if value is not None else default

    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on score."""
        if score >= 70:
            return "success"
        elif score >= 40:
            return "warning"
        return "danger"

    def _get_score_label(self, score: float) -> str:
        """Get label based on score."""
        if score >= 70:
            return "Good"
        elif score >= 40:
            return "Fair"
        return "Poor"

    def _get_change_class(self, score: float) -> str:
        """Get change class for metric cards."""
        return "positive" if score >= 50 else "negative"

    def _get_arrow_direction(self, score: float) -> str:
        """Get arrow direction based on score."""
        return "up" if score >= 50 else "down"

    def _get_gauge_color(self, score: float) -> str:
        """Get color for gauge charts."""
        if score >= 70:
            return "#28a745"
        elif score >= 40:
            return "#ffc107"
        return "#dc3545"

    def _format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%B %d, %Y at %I:%M %p")
        except:
            return timestamp

    def _interpret_distinguishability(self, auc: float) -> str:
        """Interpret distinguishability AUC score."""
        if auc > 0.8:
            return "High distinguishability - synthetic data is easily distinguishable from real data"
        elif auc > 0.6:
            return "Moderate distinguishability - some differences between synthetic and real data"
        else:
            return "Low distinguishability - synthetic data closely resembles real data"

    def _empty_section(self, name: str, message: Optional[str] = None) -> str:
        """Generate empty section placeholder."""
        detail = (
            message
            if message
            else f"No {name.lower()} data available for this experiment."
        )
        return f"""
        <div class="content-section">
            <h2 class="section-title"><i class="fas fa-exclamation-circle"></i>{name}</h2>
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                {detail}
            </div>
        </div>
        """


if __name__ == "__main__":
    # Test the generator with sample data
    import json
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            results = json.load(f)

        generator = IndividualDashboardGenerator()
        html = generator.generate(results)

        output_file = "individual_dashboard_test.html"
        with open(output_file, "w") as f:
            f.write(html)

        print(f"Dashboard generated: {output_file}")
    else:
        print("Usage: python individual_dashboard_generator.py <results.json>")
