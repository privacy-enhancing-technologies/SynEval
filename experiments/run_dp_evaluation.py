#!/usr/bin/env python3
"""
Config-driven Differential Privacy evaluation runner.

This module unifies the Stocks (time-series) and Adult (tabular) SynEval flows
behind a single YAML schema.  Both datasets describe their inputs, outputs, and
artifacts via configuration rather than hard-coded paths, enabling dashboard
generation pipelines to stay dynamic as new datasets are added.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Make repository root importable for `run.SynEval` and HTML generators.
sys.path.insert(0, str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


def _sanitize_identifier(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(raw))


def _ensure_list(value: Any) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _convert_numpy_for_json(obj: Any) -> Any:
    import numpy as _np

    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _convert_numpy_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(_convert_numpy_for_json(v) for v in obj)
    return obj


@dataclass
class DatasetConfig:
    key: str
    display_name: str
    default_columns: List[str]
    utility_target: str
    utility_inputs: List[str]
    utility_selected_metrics: Optional[List[str]] = None
    real_train_path: Optional[Path] = None
    real_test_path: Optional[Path] = None
    metadata_info_path: Optional[Path] = None
    column_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    text_columns: List[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    group_key: str
    group_display_name: str
    dataset_key: str
    epsilon: str
    synthetic_data_path: Path
    train_data_path: Path
    test_data_path: Path
    results_dir: Path
    delta: Optional[str] = None
    model_path: Optional[Path] = None
    schedule_path: Optional[Path] = None
    mia_model_path: Optional[Path] = None
    mia_adapter: Optional[str] = None
    mia_dataset_name: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    experiment_id: str = field(init=False)

    def __post_init__(self) -> None:
        eps_token = _sanitize_identifier(self.epsilon)
        if self.delta:
            delta_token = _sanitize_identifier(self.delta)
            eps_token = f"{eps_token}_delta_{delta_token}"
        self.experiment_id = f"{self.group_key}_eps_{eps_token}"


class MetricsAggregator:
    """Aggregate and compute statistics across multiple experiment results."""

    @staticmethod
    def aggregate_group_metrics(results_paths: List[Path]) -> Dict[str, Any]:
        all_results = []
        for path in results_paths:
            if path.exists():
                with open(path, "r") as handle:
                    all_results.append(json.load(handle))

        if not all_results:
            return {}

        aggregated = {
            "group": all_results[0].get("group"),
            "group_display_name": all_results[0].get("group_display_name"),
            "epsilons": [r.get("epsilon") for r in all_results],
            "experiments_count": len(all_results),
        }

        for dimension in ("fidelity", "utility", "privacy", "diversity", "mia"):
            if any(dimension in r for r in all_results):
                aggregated[dimension] = MetricsAggregator._aggregate_dimension(
                    all_results, dimension
                )
        return aggregated

    @staticmethod
    def _aggregate_dimension(results: List[Dict[str, Any]], dimension: str) -> Dict:
        dim_results = [r.get(dimension, {}) for r in results if dimension in r]
        if not dim_results:
            return {}

        keys = set()
        for r in dim_results:
            keys.update(MetricsAggregator._extract_numeric_keys(r))

        aggregated: Dict[str, Dict[str, float]] = {}
        for key in keys:
            values: List[float] = []
            for r in dim_results:
                value = MetricsAggregator._get_nested_value(r, key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if values:
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }
        return aggregated

    @staticmethod
    def _extract_numeric_keys(data: Dict[str, Any], prefix: str = "") -> List[str]:
        keys: List[str] = []
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (int, float)):
                keys.append(full_key)
            elif isinstance(value, dict):
                keys.extend(MetricsAggregator._extract_numeric_keys(value, full_key))
        return keys

    @staticmethod
    def _get_nested_value(data: Dict[str, Any], key: str) -> Optional[float]:
        cursor: Any = data
        for part in key.split("."):
            if not isinstance(cursor, dict):
                return None
            cursor = cursor.get(part)
            if cursor is None:
                return None
        if isinstance(cursor, (int, float)):
            return float(cursor)
        return None


class DashboardDPEvaluationRunner:
    """Unified orchestrator for DP dashboard generation."""

    def __init__(self, config_path: str = "configs/cross_group_dp.yaml") -> None:
        self.project_root = Path(__file__).parent.parent
        self.config_path = self._resolve_path(config_path)
        self.raw_config = self._load_config()

        self.report_cfg = self.raw_config.get("reporting", {}) or {}
        reports_dir = self.report_cfg.get("reports_output_dir", "reports/dp_evaluation")
        self.results_base_dir = self._resolve_path(reports_dir)
        self.results_base_dir.mkdir(parents=True, exist_ok=True)

        self.execution_cfg = self.raw_config.get("execution", {}) or {}
        self._configure_execution_environment()

        self.evaluations_cfg = (
            self.raw_config.get("evaluations")
            or self.raw_config.get("syneval_experiments")
            or {}
        )
        self.privacy_cfg = self.raw_config.get("privacy", {}) or {}
        self.mia_defaults = self.raw_config.get("mia", {}) or {}
        self.validation_cfg = self.raw_config.get("validation", {}) or {}

        self.datasets = self._parse_datasets()
        self.experiments = self._build_experiment_configs()

        self._data_cache: Dict[Path, pd.DataFrame] = {}
        self._dataset_overrides_cache: Dict[str, Dict[str, Any]] = {}
        self._real_data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.group_results: Dict[str, List[Dict[str, Any]]] = {}
        self.all_results: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ config
    def _resolve_path(self, path_like: str | Path) -> Path:
        path = Path(path_like)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _resolve_optional_path(self, path_like: Optional[str]) -> Optional[Path]:
        if not path_like:
            return None
        return self._resolve_path(path_like)

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r") as handle:
            config = yaml.safe_load(handle)
        logger.info("Loaded configuration from %s", self.config_path)
        return config or {}

    def _configure_execution_environment(self) -> None:
        log_level = self.execution_cfg.get("log_level")
        if isinstance(log_level, str):
            logging.getLogger().setLevel(
                getattr(logging, log_level.upper(), logging.INFO)
            )

        openblas_threads = self.execution_cfg.get("openblas_threads")
        if openblas_threads:
            thread_str = str(openblas_threads)
            os.environ.setdefault("OPENBLAS_NUM_THREADS", thread_str)
            os.environ.setdefault("OMP_NUM_THREADS", thread_str)
            os.environ.setdefault("MKL_NUM_THREADS", thread_str)

        cache_dir = self.execution_cfg.get("cache_dir")
        if self.execution_cfg.get("enable_cache") and cache_dir:
            self._resolve_path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _parse_datasets(self) -> Dict[str, DatasetConfig]:
        datasets_cfg = self.raw_config.get("datasets")
        if not datasets_cfg:
            raise ValueError("Configuration must provide a 'datasets' section.")

        datasets: Dict[str, DatasetConfig] = {}
        for dataset_key, cfg in datasets_cfg.items():
            utility_cfg = cfg.get("utility") or {}
            target = utility_cfg.get("target_column")
            inputs = utility_cfg.get("input_columns")
            if not target or not inputs:
                raise ValueError(
                    f"Dataset '{dataset_key}' must define utility.target_column and utility.input_columns."
                )

            real_cfg = cfg.get("real_data") or {}
            metadata_cfg = cfg.get("metadata") or {}
            dataset = DatasetConfig(
                key=dataset_key,
                display_name=cfg.get("display_name", dataset_key),
                default_columns=_ensure_list(cfg.get("default_columns")),
                utility_target=target,
                utility_inputs=_ensure_list(inputs),
                utility_selected_metrics=_ensure_list(
                    utility_cfg.get("selected_metrics")
                )
                or None,
                real_train_path=self._resolve_optional_path(real_cfg.get("train_path")),
                real_test_path=self._resolve_optional_path(real_cfg.get("test_path")),
                metadata_info_path=self._resolve_optional_path(
                    metadata_cfg.get("info_json_path")
                ),
                column_overrides=metadata_cfg.get("column_overrides", {}) or {},
                text_columns=_ensure_list(cfg.get("text_columns")),
            )
            datasets[dataset_key] = dataset
        return datasets

    def _build_experiment_configs(self) -> List[ExperimentConfig]:
        raw_groups = self.raw_config.get("experiment_groups")
        if not raw_groups:
            raise ValueError("Configuration must define 'experiment_groups'.")

        entries: List[Tuple[str, Dict[str, Any]]] = []
        if isinstance(raw_groups, dict):
            entries = [(key, value) for key, value in raw_groups.items()]
        elif isinstance(raw_groups, list):
            for entry in raw_groups:
                if not isinstance(entry, dict):
                    raise ValueError("experiment_groups list entries must be mappings.")
                key = entry.get("key")
                if not key:
                    raise ValueError("Each experiment group must define a 'key'.")
                entries.append((key, entry))
        else:
            raise ValueError("experiment_groups must be a mapping or list of mappings.")

        experiments: List[ExperimentConfig] = []
        for group_key, group_cfg in entries:
            dataset_key = group_cfg.get("dataset")
            if not dataset_key:
                raise ValueError(f"Group '{group_key}' missing dataset reference.")
            if dataset_key not in self.datasets:
                raise ValueError(
                    f"Group '{group_key}' references unknown dataset '{dataset_key}'."
                )

            dataset_cfg = self.datasets[dataset_key]
            display_name = group_cfg.get("display_name", group_key)
            group_mia_cfg = group_cfg.get("mia", {}) or {}

            if "generator" in group_cfg:
                experiments.extend(
                    self._expand_generator_group(
                        group_key,
                        display_name,
                        dataset_cfg,
                        group_mia_cfg,
                        group_cfg["generator"],
                    )
                )

            for exp_entry in group_cfg.get("experiments", []) or []:
                experiments.append(
                    self._build_explicit_experiment(
                        group_key,
                        display_name,
                        dataset_cfg,
                        group_mia_cfg,
                        exp_entry,
                    )
                )

        if not experiments:
            raise ValueError("No experiments were generated from the configuration.")

        return experiments

    def _expand_generator_group(
        self,
        group_key: str,
        group_display_name: str,
        dataset_cfg: DatasetConfig,
        group_mia_cfg: Dict[str, Any],
        generator_cfg: Dict[str, Any],
    ) -> List[ExperimentConfig]:
        base_dir_raw = generator_cfg.get("base_dir")
        epsilons = generator_cfg.get("epsilons")
        if not base_dir_raw or not epsilons:
            raise ValueError(
                f"Generator for group '{group_key}' must define 'base_dir' and 'epsilons'."
            )

        base_dir = self._resolve_path(base_dir_raw)
        epsilon_prefix = generator_cfg.get("epsilon_prefix", "")
        data_cfg = generator_cfg.get("data", {}) or {}
        model_cfg = generator_cfg.get("model", {}) or {}

        data_subdir = data_cfg.get("subdir", "")
        synthetic_file = data_cfg.get("synthetic")
        train_file = data_cfg.get("train")
        test_file = data_cfg.get("test")
        if not synthetic_file:
            raise ValueError(
                f"Generator for group '{group_key}' must provide data.synthetic."
            )

        experiments: List[ExperimentConfig] = []
        for epsilon in _ensure_list(epsilons):
            exp_dir = base_dir / f"{epsilon_prefix}{epsilon}"
            data_root = exp_dir / data_subdir if data_subdir else exp_dir

            synthetic_path = data_root / synthetic_file
            train_path = (
                data_root / train_file if train_file else dataset_cfg.real_train_path
            )
            test_path = (
                data_root / test_file if test_file else dataset_cfg.real_test_path
            )

            if train_path is None or test_path is None:
                raise ValueError(
                    f"Group '{group_key}' is missing train/test paths for epsilon {epsilon}."
                )

            model_path: Optional[Path] = None
            schedule_path: Optional[Path] = None
            if "file" in model_cfg:
                model_root = exp_dir / model_cfg.get("subdir", "")
                model_path = model_root / model_cfg["file"]
            elif "files" in model_cfg:
                model_root = exp_dir / model_cfg.get("subdir", "")
                files_dict = model_cfg.get("files") or {}
                if "primary" in files_dict:
                    model_path = model_root / files_dict["primary"]
                if "schedule" in files_dict:
                    schedule_path = model_root / files_dict["schedule"]

            results_dir = (
                self.results_base_dir
                / group_key
                / f"eps_{_sanitize_identifier(epsilon)}"
            )
            results_dir.mkdir(parents=True, exist_ok=True)

            experiments.append(
                ExperimentConfig(
                    group_key=group_key,
                    group_display_name=group_display_name,
                    dataset_key=dataset_cfg.key,
                    epsilon=str(epsilon),
                    synthetic_data_path=synthetic_path,
                    train_data_path=train_path,
                    test_data_path=test_path,
                    results_dir=results_dir,
                    model_path=model_path,
                    schedule_path=schedule_path,
                    mia_adapter=group_mia_cfg.get("adapter")
                    or self.mia_defaults.get("adapter"),
                    mia_dataset_name=group_mia_cfg.get("dataset")
                    or self.mia_defaults.get("dataset")
                    or dataset_cfg.key,
                )
            )
        return experiments

    def _build_explicit_experiment(
        self,
        group_key: str,
        group_display_name: str,
        dataset_cfg: DatasetConfig,
        group_mia_cfg: Dict[str, Any],
        entry: Dict[str, Any],
    ) -> ExperimentConfig:
        epsilon = entry.get("epsilon")
        if epsilon is None:
            raise ValueError(
                f"Explicit experiment for group '{group_key}' missing epsilon."
            )

        synthetic_raw = entry.get("synthetic_path")
        if not synthetic_raw:
            raise ValueError(
                f"Explicit experiment for group '{group_key}' (eps={epsilon}) missing synthetic_path."
            )

        train_raw = entry.get("train_path")
        test_raw = entry.get("test_path")
        train_path = (
            self._resolve_optional_path(train_raw) or dataset_cfg.real_train_path
        )
        test_path = self._resolve_optional_path(test_raw) or dataset_cfg.real_test_path
        if train_path is None or test_path is None:
            raise ValueError(
                f"Explicit experiment for group '{group_key}' (eps={epsilon}) requires train/test paths."
            )

        synthetic_path = self._resolve_path(synthetic_raw)
        delta = entry.get("delta")

        model_paths = entry.get("model_paths", {}) or {}
        model_path = self._resolve_optional_path(model_paths.get("primary"))
        schedule_path = self._resolve_optional_path(model_paths.get("schedule"))
        mia_model_path = self._resolve_optional_path(model_paths.get("mia"))

        results_token = _sanitize_identifier(str(epsilon))
        if delta:
            results_token = f"{results_token}_delta_{_sanitize_identifier(str(delta))}"
        results_dir = self.results_base_dir / group_key / f"eps_{results_token}"
        results_dir.mkdir(parents=True, exist_ok=True)

        mia_cfg = entry.get("mia", {}) or {}
        mia_adapter = (
            mia_cfg.get("adapter")
            or group_mia_cfg.get("adapter")
            or self.mia_defaults.get("adapter")
        )
        mia_dataset_name = (
            mia_cfg.get("dataset")
            or group_mia_cfg.get("dataset")
            or self.mia_defaults.get("dataset")
            or dataset_cfg.key
        )

        return ExperimentConfig(
            group_key=group_key,
            group_display_name=group_display_name,
            dataset_key=dataset_cfg.key,
            epsilon=str(epsilon),
            delta=str(delta) if delta is not None else None,
            synthetic_data_path=synthetic_path,
            train_data_path=train_path,
            test_data_path=test_path,
            results_dir=results_dir,
            model_path=model_path,
            schedule_path=schedule_path,
            mia_model_path=mia_model_path,
            mia_adapter=mia_adapter,
            mia_dataset_name=mia_dataset_name,
            extra_metadata={"delta": delta} if delta is not None else {},
        )

    # ----------------------------------------------------------------- helpers
    def _validate_experiment_files(
        self, exp_config: ExperimentConfig
    ) -> Tuple[bool, List[str]]:
        if not self.validation_cfg.get("check_files_exist", True):
            return True, []

        missing: List[str] = []
        if not exp_config.synthetic_data_path.exists():
            missing.append(f"Synthetic data: {exp_config.synthetic_data_path}")
        if not exp_config.train_data_path.exists():
            missing.append(f"Train data: {exp_config.train_data_path}")
        if not exp_config.test_data_path.exists():
            missing.append(f"Test data: {exp_config.test_data_path}")
        return (not missing, missing)

    def _read_dataframe(
        self,
        path: Path,
        dataset_cfg: DatasetConfig,
    ) -> pd.DataFrame:
        absolute = path if path.is_absolute() else self._resolve_path(path)
        if absolute in self._data_cache:
            return self._data_cache[absolute].copy()

        if absolute.suffix.lower() == ".csv":
            df = pd.read_csv(absolute)
        elif absolute.suffix.lower() == ".json":
            df = pd.read_json(absolute)
        elif absolute.suffix.lower() == ".npy":
            array = np.load(absolute, allow_pickle=True)
            if array.ndim == 3:
                array = array.reshape(-1, array.shape[-1])
            n_features = array.shape[1] if array.ndim > 1 else 1
            columns = dataset_cfg.default_columns[:n_features]
            df = pd.DataFrame(array, columns=columns)
        else:
            raise ValueError(f"Unsupported data format: {absolute}")

        df = self._rename_default_columns(df, dataset_cfg)
        self._data_cache[absolute] = df
        return df.copy()

    def _rename_default_columns(
        self, df: pd.DataFrame, dataset_cfg: DatasetConfig
    ) -> pd.DataFrame:
        renamed = df.copy()
        digit_like = all(str(col).isdigit() for col in renamed.columns)
        if digit_like and dataset_cfg.default_columns:
            rename_map: Dict[str, str] = {}
            for col in renamed.columns:
                idx = int(col)
                if 0 <= idx < len(dataset_cfg.default_columns):
                    rename_map[col] = dataset_cfg.default_columns[idx]
            if rename_map:
                renamed = renamed.rename(columns=rename_map)
        return renamed

    def _coerce_utility_columns(
        self,
        df: pd.DataFrame,
        dataset_cfg: DatasetConfig,
    ) -> pd.DataFrame:
        coerced = df.copy()
        for col in dataset_cfg.utility_inputs:
            if col in coerced.columns:
                coerced[col] = pd.to_numeric(coerced[col], errors="coerce")
        target = dataset_cfg.utility_target
        if target in coerced.columns and coerced[target].dtype == object:
            coerced[target] = coerced[target].astype(str).str.strip()
        return coerced

    def _load_data(
        self,
        exp_config: ExperimentConfig,
        dataset_cfg: DatasetConfig,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        synthetic = self._read_dataframe(exp_config.synthetic_data_path, dataset_cfg)
        train = self._read_dataframe(exp_config.train_data_path, dataset_cfg)
        test = self._read_dataframe(exp_config.test_data_path, dataset_cfg)

        reference_columns = train.columns.tolist()
        for name, df in (("synthetic", synthetic), ("test", test)):
            missing = [col for col in reference_columns if col not in df.columns]
            if missing:
                raise ValueError(
                    f"{name.capitalize()} data for {exp_config.experiment_id} missing columns: {missing}"
                )

        synthetic = synthetic[reference_columns]
        test = test[reference_columns]

        synthetic = self._coerce_utility_columns(synthetic, dataset_cfg)
        train = self._coerce_utility_columns(train, dataset_cfg)
        test = self._coerce_utility_columns(test, dataset_cfg)

        target = dataset_cfg.utility_target
        required = dataset_cfg.utility_inputs + [target]
        missing_inputs = [col for col in required if col not in synthetic.columns]
        if missing_inputs:
            raise ValueError(
                f"Utility columns missing for {exp_config.experiment_id}: {missing_inputs}"
            )

        return train, test, synthetic

    def _load_dataset_column_overrides(
        self, dataset_cfg: DatasetConfig
    ) -> Dict[str, Dict[str, Any]]:
        if dataset_cfg.key in self._dataset_overrides_cache:
            return self._dataset_overrides_cache[dataset_cfg.key]

        overrides = dataset_cfg.column_overrides.copy()
        info_path = dataset_cfg.metadata_info_path
        if info_path and info_path.exists():
            try:
                with open(info_path, "r") as handle:
                    info_json = json.load(handle)
                column_names = info_json.get("column_names", [])
                column_info = info_json.get("column_info", {})
                for idx, name in enumerate(column_names):
                    details = column_info.get(str(idx), {})
                    col_type = str(details.get("type", "")).lower()
                    if col_type == "numerical":
                        overrides.setdefault(name, {"sdtype": "numerical"})
                    elif col_type == "categorical":
                        categories = details.get("categorizes")
                        if categories:
                            categories = [str(cat).strip() for cat in categories]
                        overrides.setdefault(
                            name, {"sdtype": "categorical", "categories": categories}
                        )
            except Exception as exc:
                logger.warning(
                    "Failed to interpret metadata info JSON for dataset %s: %s",
                    dataset_cfg.key,
                    exc,
                )
        self._dataset_overrides_cache[dataset_cfg.key] = overrides
        return overrides

    def _generate_metadata(
        self,
        data: pd.DataFrame,
        dataset_cfg: DatasetConfig,
    ) -> Dict[str, Any]:
        overrides = self._load_dataset_column_overrides(dataset_cfg)
        metadata: Dict[str, Any] = {
            "columns": {},
            "primary_key": None,
            "METADATA_SPEC_VERSION": "1.0",
        }

        for col in data.columns:
            if col in overrides:
                col_meta = dict(overrides[col])
            else:
                dtype = data[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    col_meta = {"sdtype": "numerical"}
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    col_meta = {"sdtype": "datetime"}
                else:
                    col_meta = {"sdtype": "categorical"}

            if col_meta.get("sdtype") == "categorical" and "categories" not in col_meta:
                unique = (
                    data[col].dropna().unique().tolist() if not data[col].empty else []
                )
                col_meta["categories"] = [str(val) for val in unique][:200]
            metadata["columns"][col] = col_meta
        return metadata

    # --------------------------------------------------------------- reporting
    def _call_individual_report_generator(
        self,
        exp_config: ExperimentConfig,
        results: Dict[str, Any],
    ) -> None:
        if not self.report_cfg.get("individual_reports", True):
            return
        try:
            from experiments.utils.dp_html_generator import \
                DPHtmlGenerator  # type: ignore

            generator = DPHtmlGenerator(output_dir=exp_config.results_dir)
            output_path = generator.generate_individual_dashboard(
                results=results,
                experiment_name=exp_config.group_display_name,
            )
            logger.info(
                "Saved individual dashboard for %s to %s",
                exp_config.experiment_id,
                output_path,
            )
        except ImportError as exc:
            logger.warning("Individual HTML generator unavailable: %s", exc)
        except Exception as exc:
            logger.error(
                "Failed to generate individual dashboard for %s: %s",
                exp_config.experiment_id,
                exc,
            )

    def _call_in_group_report_generator(
        self,
        group_key: str,
        group_display_name: str,
        group_results: List[Dict[str, Any]],
    ) -> None:
        if not self.report_cfg.get("in_group_comparisons", True):
            return
        try:
            from experiments.utils.dp_html_generator import \
                DPHtmlGenerator  # type: ignore
        except ImportError:
            try:
                from dp_html_generator import DPHtmlGenerator  # type: ignore
            except ImportError:
                logger.warning("HTML generator unavailable; skipping in-group report.")
                return
        successful = [res for res in group_results if res.get("status") == "success"]
        if not successful:
            logger.warning(
                "Skipping in-group report for %s (no successful runs).", group_key
            )
            return
        output_dir = self.results_base_dir / group_key / "in_group_comparison"
        try:
            generator = DPHtmlGenerator(output_dir=output_dir)
            output_path = generator.generate_in_group_comparison(
                group_results=successful,
                group_name=group_display_name,
            )
            if output_path:
                logger.info(
                    "Saved in-group dashboard for %s to %s",
                    group_key,
                    output_path,
                )
        except Exception as exc:
            logger.error(
                "Failed to generate in-group report for %s: %s", group_key, exc
            )

    def _call_cross_group_report_generator(self) -> None:
        if not self.report_cfg.get("cross_group_comparisons", True):
            return
        try:
            from experiments.utils.dp_html_generator import \
                DPHtmlGenerator  # type: ignore
        except ImportError:
            try:
                from dp_html_generator import DPHtmlGenerator  # type: ignore
            except ImportError:
                logger.warning(
                    "HTML generator unavailable; skipping cross-group report."
                )
                return
        try:
            output_dir = self.results_base_dir / "cross_group_comparison"
            generator = DPHtmlGenerator(output_dir=output_dir)
            output_path = generator.generate_cross_group_comparison(
                all_results=self.group_results
            )
            logger.info("Saved cross-group dashboard to %s", output_path)
        except Exception as exc:
            logger.error("Failed to generate cross-group report: %s", exc)

    # -------------------------------------------------------------- execution
    def _run_single_experiment(self, exp_config: ExperimentConfig) -> Dict[str, Any]:
        dataset_cfg = self.datasets[exp_config.dataset_key]
        valid, missing = self._validate_experiment_files(exp_config)
        if not valid:
            error_msg = "; ".join(missing)
            logger.error(
                "Missing files for %s: %s", exp_config.experiment_id, error_msg
            )
            return {
                "status": "failed",
                "error": error_msg,
                "experiment_id": exp_config.experiment_id,
            }

        try:
            train_data, test_data, synthetic_data = self._load_data(
                exp_config, dataset_cfg
            )
        except Exception as exc:
            logger.error(
                "Data loading failed for %s: %s", exp_config.experiment_id, exc
            )
            return {
                "status": "failed",
                "error": str(exc),
                "experiment_id": exp_config.experiment_id,
            }

        real_data = pd.concat([train_data, test_data], ignore_index=True)
        if self.privacy_cfg.get("max_rows"):
            max_rows = int(self.privacy_cfg["max_rows"])
            real_data = real_data.sample(
                n=min(len(real_data), max_rows), random_state=42
            ).reset_index(drop=True)
            synthetic_data = synthetic_data.sample(
                n=min(len(synthetic_data), max_rows),
                random_state=42,
            ).reset_index(drop=True)

        metadata = self._generate_metadata(real_data, dataset_cfg)
        metadata_path = exp_config.results_dir / "metadata.json"
        with open(metadata_path, "w") as handle:
            json.dump(metadata, handle, indent=2)
        logger.info("Saved metadata to %s", metadata_path)

        from run import SynEval  # Deferred import

        device = self.execution_cfg.get("device", "auto")
        syneval = SynEval(
            synthetic_data,
            real_data,
            metadata,
            device=device,
        )

        results: Dict[str, Any] = {
            "experiment_id": exp_config.experiment_id,
            "group": exp_config.group_key,
            "group_display_name": exp_config.group_display_name,
            "epsilon": exp_config.epsilon,
            "delta": exp_config.delta,
            "status": "success",
            "metadata_path": str(metadata_path),
            "synthetic_path": str(exp_config.synthetic_data_path),
            "train_path": str(exp_config.train_data_path),
            "test_path": str(exp_config.test_data_path),
        }
        results.update(exp_config.extra_metadata)

        failure = False

        # Fidelity
        fidelity_cfg = (
            (self.evaluations_cfg.get("fidelity") or {}) if self.evaluations_cfg else {}
        )
        if fidelity_cfg.get("enabled", True):
            try:
                selected = fidelity_cfg.get("selected_metrics")
                results["fidelity"] = syneval.evaluate_fidelity(
                    selected_metrics=selected
                )
            except Exception as exc:
                logger.error(
                    "Fidelity evaluation failed for %s: %s",
                    exp_config.experiment_id,
                    exc,
                )
                results["fidelity"] = {"status": "failed", "error": str(exc)}
                failure = True

        # Utility
        utility_cfg = (
            (self.evaluations_cfg.get("utility") or {}) if self.evaluations_cfg else {}
        )
        if utility_cfg.get("enabled", True):
            try:
                selected = dataset_cfg.utility_selected_metrics or utility_cfg.get(
                    "selected_metrics"
                )
                results["utility"] = syneval.evaluate_utility(
                    input_columns=dataset_cfg.utility_inputs,
                    output_columns=[dataset_cfg.utility_target],
                    selected_metrics=selected,
                    real_train_data=train_data,
                    real_test_data=test_data,
                )
            except Exception as exc:
                logger.error(
                    "Utility evaluation failed for %s: %s",
                    exp_config.experiment_id,
                    exc,
                )
                results["utility"] = {"status": "failed", "error": str(exc)}
                failure = True

        # Privacy
        privacy_cfg = (
            (self.evaluations_cfg.get("privacy") or {}) if self.evaluations_cfg else {}
        )
        if privacy_cfg.get("enabled", True):
            privacy_metrics = privacy_cfg.get("selected_metrics") or [
                "exact_matches",
                "tabular_privacy",
                "text_privacy",
            ]
            if (
                self.privacy_cfg.get("skip_anonymeter", True)
                and "anonymeter" in privacy_metrics
            ):
                privacy_metrics.remove("anonymeter")
            try:
                results["privacy"] = syneval.evaluate_privacy(
                    selected_metrics=privacy_metrics
                )
            except Exception as exc:
                logger.error(
                    "Privacy evaluation failed for %s: %s",
                    exp_config.experiment_id,
                    exc,
                )
                results["privacy"] = {
                    "status": "failed",
                    "error": str(exc),
                    "selected_metrics": privacy_metrics,
                }
                failure = True

        # Diversity
        diversity_cfg = (
            (self.evaluations_cfg.get("diversity") or {})
            if self.evaluations_cfg
            else {}
        )
        if diversity_cfg.get("enabled", True):
            try:
                selected = diversity_cfg.get("selected_metrics")
                results["diversity"] = syneval.evaluate_diversity(
                    selected_metrics=selected
                )
            except Exception as exc:
                logger.error(
                    "Diversity evaluation failed for %s: %s",
                    exp_config.experiment_id,
                    exc,
                )
                results["diversity"] = {"status": "failed", "error": str(exc)}
                failure = True

        # MIA
        mia_cfg = (
            (self.evaluations_cfg.get("mia") or {}) if self.evaluations_cfg else {}
        )
        if mia_cfg.get("enabled", True):
            try:
                from experiments.utils.membership_inference import \
                    run_mia_for_experiment  # type: ignore
            except ImportError as exc:
                logger.info("MIA module not available, skipping: %s", exc)
                results["mia"] = {
                    "status": "skipped",
                    "reason": "membership_inference module not available",
                }
            else:
                try:
                    mia_dataset = exp_config.mia_dataset_name or self.mia_defaults.get(
                        "dataset"
                    )
                    adapter_name = exp_config.mia_adapter or self.mia_defaults.get(
                        "adapter"
                    )
                    n_shadow = int(
                        mia_cfg.get("n_shadow")
                        or self.mia_defaults.get("n_shadow", 1000)
                    )
                    mia_result = run_mia_for_experiment(
                        model_path=exp_config.model_path,
                        schedule_path=exp_config.schedule_path,
                        train_data_path=exp_config.train_data_path,
                        test_data_path=exp_config.test_data_path,
                        n_shadow=n_shadow,
                        dataset_name=mia_dataset,
                        adapter_name=adapter_name,
                    )
                    results["mia"] = mia_result
                except Exception as exc:
                    logger.error(
                        "MIA evaluation failed for %s: %s",
                        exp_config.experiment_id,
                        exc,
                    )
                    results["mia"] = {"status": "failed", "error": str(exc)}
                    failure = True

        results_path = exp_config.results_dir / "results.json"
        with open(results_path, "w") as handle:
            json.dump(_convert_numpy_for_json(results), handle, indent=2)
        logger.info("Saved experiment results to %s", results_path)

        if failure:
            results["status"] = "partial_failure"
        return results

    def _process_experiments(self, experiments: Iterable[ExperimentConfig]) -> None:
        for exp_config in experiments:
            logger.info(
                "Running experiment %s (ε=%s)",
                exp_config.experiment_id,
                exp_config.epsilon,
            )
            result = self._run_single_experiment(exp_config)
            self.all_results[exp_config.experiment_id] = result
            self.group_results.setdefault(exp_config.group_key, []).append(result)

        for group_key, results in self.group_results.items():
            display_name = next(
                (
                    exp.group_display_name
                    for exp in self.experiments
                    if exp.group_key == group_key
                ),
                group_key,
            )
            self._call_in_group_report_generator(group_key, display_name, results)
        self._call_cross_group_report_generator()

    def run_sequential(self) -> None:
        logger.info(
            "Starting sequential execution of %d experiments", len(self.experiments)
        )
        self._process_experiments(self.experiments)

    def run_parallel(self, max_workers: int = 4) -> None:
        logger.warning(
            "Parallel execution is temporarily mapped to sequential processing in the unified runner."
        )
        self._process_experiments(self.experiments)

    def save_summary(self) -> Path:
        summary = {
            "config": str(self.config_path),
            "total_experiments": len(self.experiments),
            "successful": sum(
                1 for r in self.all_results.values() if r.get("status") == "success"
            ),
            "failed": sum(
                1 for r in self.all_results.values() if r.get("status") == "failed"
            ),
        }
        summary_path = self.results_base_dir / "summary.json"
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Summary written to %s", summary_path)
        return summary_path


def main(default_config: str = "configs/cross_group_dp.yaml") -> int:
    parser = argparse.ArgumentParser(description="Run SynEval DP dashboard pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to unified DP dashboard configuration file",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel (ProcessPoolExecutor)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers to use with --parallel",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    runner = DashboardDPEvaluationRunner(config_path=args.config)
    if args.parallel:
        runner.run_parallel(max_workers=args.workers)
    else:
        runner.run_sequential()
    runner.save_summary()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Execution failed: %s\n%s", exc, traceback.format_exc())
        sys.exit(1)
