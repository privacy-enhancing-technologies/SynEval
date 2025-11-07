# experiments/utils/dp_html_generator.py
#!/usr/bin/env python3
"""
Differential Privacy HTML Dashboard Generator
=============================================
Main orchestrator for generating beautiful, interactive HTML dashboards
for DP evaluation results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

# Import specialized generators
try:
    from .individual_dashboard_generator import IndividualDashboardGenerator
except ImportError:
    # Handle absolute imports when running as script
    from individual_dashboard_generator import IndividualDashboardGenerator

# In-group and cross-group generators are optional (still in development)
try:
    from .in_group_dashboard_generator import InGroupDashboardGenerator
except ImportError:
    try:
        from in_group_dashboard_generator import InGroupDashboardGenerator
    except ImportError:
        InGroupDashboardGenerator = None  # type: ignore

try:
    from .cross_group_dashboard_generator import CrossGroupDashboardGenerator
except ImportError:
    try:
        from cross_group_dashboard_generator import \
            CrossGroupDashboardGenerator
    except ImportError:
        CrossGroupDashboardGenerator = None  # type: ignore

logger = logging.getLogger(__name__)


class DPHtmlGenerator:
    """
    Main class for generating HTML dashboards from DP evaluation results.
    Coordinates the creation of individual, in-group, and cross-group dashboards.
    """

    def __init__(self, output_dir: Union[str, Path] = "reports/dp_evaluation"):
        """
        Initialize the HTML generator.

        Args:
            output_dir: Directory where HTML reports will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize specialized generators
        self.individual_gen = IndividualDashboardGenerator()
        self.in_group_gen = None  # Will be initialized when needed
        self.cross_group_gen = None  # Will be initialized when needed

        # Store metadata
        self.generation_timestamp = datetime.now().isoformat()
        self.generated_reports = []

        logger.info(
            f"Initialized DP HTML Generator with output directory: {self.output_dir}"
        )

    def generate_individual_dashboard(
        self,
        results: Dict,
        output_file: Optional[Path] = None,
        experiment_name: Optional[str] = None,
    ) -> Path:
        """
        Generate an individual experiment dashboard.

        Args:
            results: Dictionary containing experiment results
            output_file: Optional custom output file path
            experiment_name: Optional experiment name for display

        Returns:
            Path to generated HTML file
        """
        if output_file is None:
            # Generate default filename based on experiment ID
            exp_id = results.get("experiment_id", "unknown")
            output_file = self.output_dir / f"individual_{exp_id}.html"
        else:
            output_file = Path(output_file)

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Generate the dashboard
        logger.info(
            f"Generating individual dashboard for {experiment_name or results.get('experiment_id', 'unknown')}"
        )

        html_content = self.individual_gen.generate(
            results=results, experiment_name=experiment_name
        )

        # Write to file
        output_file.write_text(html_content)

        # Track generated report
        self.generated_reports.append(
            {
                "type": "individual",
                "path": str(output_file),
                "experiment": experiment_name or results.get("experiment_id"),
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(f"✅ Individual dashboard saved to: {output_file}")
        return output_file

    def generate_in_group_comparison(
        self,
        group_results: List[Dict],
        group_name: str,
        output_file: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate an in-group comparison dashboard (same experiment, different epsilons).

        Args:
            group_results: List of result dictionaries for the group
            group_name: Name of the experiment group
            output_file: Optional custom output file path

        Returns:
            Path to generated HTML file
        """
        if not group_results:
            logger.warning(
                "No successful results provided for in-group comparison '%s'; skipping generation",
                group_name,
            )
            return None

        if self.in_group_gen is None:
            try:
                from .in_group_dashboard_generator import \
                    InGroupDashboardGenerator
            except ImportError:
                from in_group_dashboard_generator import \
                    InGroupDashboardGenerator
            self.in_group_gen = InGroupDashboardGenerator()

        if output_file is None:
            # Generate default filename based on group name
            safe_name = group_name.replace(" ", "_").replace(",", "").lower()
            output_file = self.output_dir / f"in_group_{safe_name}.html"
        else:
            output_file = Path(output_file)

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Generate the dashboard
        logger.info(f"Generating in-group comparison for {group_name}")

        html_content = self.in_group_gen.generate(
            group_results=group_results, group_name=group_name
        )

        # Write to file
        output_file.write_text(html_content)

        # Track generated report
        self.generated_reports.append(
            {
                "type": "in_group",
                "path": str(output_file),
                "group": group_name,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(f"✅ In-group comparison dashboard saved to: {output_file}")
        return output_file

    def generate_cross_group_comparison(
        self,
        all_results: Dict[str, List[Dict]],
        output_file: Optional[Path] = None,
        comparison_name: str = "Cross-Group DP Comparison",
    ) -> Path:
        """
        Generate a cross-group comparison dashboard.

        Args:
            all_results: Dictionary mapping group names to lists of results
            output_file: Optional custom output file path
            comparison_name: Name for the comparison

        Returns:
            Path to generated HTML file
        """
        if self.cross_group_gen is None:
            try:
                from .cross_group_dashboard_generator import \
                    CrossGroupDashboardGenerator
            except ImportError:
                from cross_group_dashboard_generator import \
                    CrossGroupDashboardGenerator
            self.cross_group_gen = CrossGroupDashboardGenerator()

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"cross_group_comparison_{timestamp}.html"
        else:
            output_file = Path(output_file)

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Generate the dashboard
        logger.info(f"Generating cross-group comparison: {comparison_name}")

        html_content = self.cross_group_gen.generate(
            all_results=all_results, comparison_name=comparison_name
        )

        # Write to file
        output_file.write_text(html_content)

        # Track generated report
        self.generated_reports.append(
            {
                "type": "cross_group",
                "path": str(output_file),
                "name": comparison_name,
                "groups": list(all_results.keys()),
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(f"✅ Cross-group comparison dashboard saved to: {output_file}")
        return output_file

    def generate_index_page(self) -> Path:
        """
        Generate an index page linking to all generated dashboards.

        Returns:
            Path to generated index HTML file
        """
        index_file = self.output_dir / "index.html"

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DP Evaluation Dashboard Index</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {{
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --card-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        
        .header {{
            background: var(--primary-gradient);
            color: white;
            padding: 3rem 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .dashboard-card {{
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--card-shadow);
            transition: all 0.3s ease;
        }}
        
        .dashboard-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.15);
        }}
        
        .badge-type {{
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .type-individual {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; }}
        .type-in-group {{ background: linear-gradient(135deg, #f093fb, #f5576c); color: white; }}
        .type-cross-group {{ background: linear-gradient(135deg, #4facfe, #00f2fe); color: white; }}
        
        .btn-view {{
            background: var(--primary-gradient);
            color: white;
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 25px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        
        .btn-view:hover {{
            transform: scale(1.05);
            color: white;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .timestamp {{
            color: #6c757d;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1 class="display-4 mb-3">
                <i class="fas fa-chart-line me-3"></i>
                Differential Privacy Evaluation Dashboards
            </h1>
            <p class="lead mb-0">
                Generated on {self.generation_timestamp}
            </p>
        </div>
    </div>
    
    <div class="container my-5">
        <div class="row">
            <div class="col-lg-12">
                <h2 class="mb-4">Available Dashboards</h2>
                
                {self._generate_dashboard_cards()}
                
                {self._generate_summary_stats()}
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

        index_file.write_text(html_content)
        logger.info(f"✅ Index page saved to: {index_file}")
        return index_file

    def _generate_dashboard_cards(self) -> str:
        """Generate HTML cards for each dashboard."""
        if not self.generated_reports:
            return """
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                No dashboards have been generated yet.
            </div>
            """

        cards_html = ""
        for report in self.generated_reports:
            report_path = Path(report["path"])
            relative_path = (
                report_path.relative_to(self.output_dir)
                if report_path.is_relative_to(self.output_dir)
                else report_path.name
            )

            type_class = f"type-{report['type'].replace('_', '-')}"

            cards_html += f"""
            <div class="dashboard-card">
                <div class="row align-items-center">
                    <div class="col-md-8">
                        <div class="d-flex align-items-center mb-2">
                            <span class="badge-type {type_class} me-3">
                                {report['type'].replace('_', ' ').title()}
                            </span>
                            <span class="timestamp">
                                <i class="far fa-clock me-1"></i>
                                {report['timestamp']}
                            </span>
                        </div>
                        <h4>{self._get_report_title(report)}</h4>
                        {self._get_report_details(report)}
                    </div>
                    <div class="col-md-4 text-end">
                        <a href="{relative_path}" class="btn btn-view">
                            <i class="fas fa-external-link-alt me-2"></i>
                            View Dashboard
                        </a>
                    </div>
                </div>
            </div>
            """

        return cards_html

    def _get_report_title(self, report: Dict) -> str:
        """Get a formatted title for the report."""
        if report["type"] == "individual":
            return f"Experiment: {report.get('experiment', 'Unknown')}"
        elif report["type"] == "in_group":
            return f"Group: {report.get('group', 'Unknown')}"
        elif report["type"] == "cross_group":
            return report.get("name", "Cross-Group Comparison")
        return "Unknown Report"

    def _get_report_details(self, report: Dict) -> str:
        """Get additional details for the report."""
        if report["type"] == "cross_group" and "groups" in report:
            groups_list = ", ".join(report["groups"][:3])
            if len(report["groups"]) > 3:
                groups_list += f" and {len(report['groups']) - 3} more"
            return f"<p class='mb-0 text-muted'>Comparing: {groups_list}</p>"
        return ""

    def _generate_summary_stats(self) -> str:
        """Generate summary statistics."""
        if not self.generated_reports:
            return ""

        counts = {
            "individual": sum(
                1 for r in self.generated_reports if r["type"] == "individual"
            ),
            "in_group": sum(
                1 for r in self.generated_reports if r["type"] == "in_group"
            ),
            "cross_group": sum(
                1 for r in self.generated_reports if r["type"] == "cross_group"
            ),
        }

        return f"""
        <div class="row mt-5">
            <div class="col-md-4">
                <div class="card text-center">
                    <div class="card-body">
                        <h3 class="mb-0">{counts['individual']}</h3>
                        <p class="text-muted">Individual Dashboards</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card text-center">
                    <div class="card-body">
                        <h3 class="mb-0">{counts['in_group']}</h3>
                        <p class="text-muted">In-Group Comparisons</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card text-center">
                    <div class="card-body">
                        <h3 class="mb-0">{counts['cross_group']}</h3>
                        <p class="text-muted">Cross-Group Comparisons</p>
                    </div>
                </div>
            </div>
        </div>
        """


def main():
    """Main function for testing the generator."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dp_html_generator.py <results_json_file>")
        sys.exit(1)

    # Load results
    results_file = Path(sys.argv[1])
    with open(results_file, "r") as f:
        results = json.load(f)

    # Initialize generator
    generator = DPHtmlGenerator()

    # Generate individual dashboard
    output_file = generator.generate_individual_dashboard(results)
    print(f"Dashboard generated: {output_file}")

    # Generate index
    index_file = generator.generate_index_page()
    print(f"Index page generated: {index_file}")


if __name__ == "__main__":
    main()
