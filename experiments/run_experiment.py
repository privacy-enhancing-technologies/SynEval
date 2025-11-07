#!/usr/bin/env python3
"""
Run a single SynEval experiment and generate its individual dashboard.

Example:
    python experiments/run_experiment.py \
        --config configs/cross_group_dp.yaml \
        --group OUTPUT_GRID_ROW_DP_EXTREMA \
        --epsilon 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repository root is importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import Optional

from experiments.run_dp_evaluation import \
    DashboardDPEvaluationRunner  # type: ignore

logger = logging.getLogger(__name__)


def _pick_experiment(
    runner: DashboardDPEvaluationRunner,
    group_key: str,
    epsilon: str,
    delta: Optional[str],
) -> Optional[object]:
    candidates = [
        exp
        for exp in runner.experiments
        if exp.group_key == group_key
        and exp.epsilon == str(epsilon)
        and (delta is None or exp.delta == delta)
    ]
    if not candidates:
        return None
    if len(candidates) > 1 and delta is None:
        logger.warning(
            "Multiple experiments match group=%s, epsilon=%s. "
            "Provide --delta to disambiguate (taking first match for now).",
            group_key,
            epsilon,
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single DP dashboard experiment and generate its individual report."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cross_group_dp.yaml",
        help="Path to unified DP configuration file.",
    )
    parser.add_argument(
        "--group",
        required=True,
        help="Experiment group key (see experiment_groups[].key in the config).",
    )
    parser.add_argument(
        "--epsilon",
        required=True,
        help="Epsilon value to run.",
    )
    parser.add_argument(
        "--delta",
        default=None,
        help="Optional delta value to disambiguate experiments when multiple entries share the same epsilon.",
    )
    parser.add_argument(
        "--disable-mia",
        action="store_true",
        help="Skip membership inference evaluation for this run.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    runner = DashboardDPEvaluationRunner(config_path=args.config)
    if args.disable_mia:
        runner.evaluations_cfg.setdefault("mia", {})["enabled"] = False

    experiment = _pick_experiment(runner, args.group, args.epsilon, args.delta)
    if not experiment:
        logger.error(
            "No experiment found for group=%s epsilon=%s delta=%s",
            args.group,
            args.epsilon,
            args.delta,
        )
        return 1

    logger.info(
        "Running experiment %s (ε=%s%s) defined in %s",
        experiment.experiment_id,
        experiment.epsilon,
        f", δ={experiment.delta}" if experiment.delta else "",
        Path(args.config),
    )

    result = runner._run_single_experiment(
        experiment
    )  # pylint: disable=protected-access
    runner.all_results[experiment.experiment_id] = result

    # Generate the individual dashboard only.
    runner._call_individual_report_generator(
        experiment, result
    )  # pylint: disable=protected-access

    status = result.get("status")
    logger.info("Experiment finished with status: %s", status)
    return 0 if status in {"success", "partial_failure"} else 1


if __name__ == "__main__":
    sys.exit(main())
