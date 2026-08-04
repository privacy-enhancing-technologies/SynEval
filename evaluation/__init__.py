"""
Evaluation module for SynEval framework.

This module contains all evaluation components:
- UtilityEvaluator: Evaluates utility metrics
- FidelityEvaluator: Evaluates fidelity metrics
- PrivacyEvaluator: Evaluates privacy metrics
- DiversityEvaluator: Evaluates diversity metrics
- MultimodalEvaluator: Unified multimodal evaluation interface
- auto_detect_columns: Auto-detect text vs tabular columns
- Multimodal evaluation functions for each dimension
"""

from .column_detector import auto_detect_columns
from .diversity import DiversityEvaluator, compute_joint_entropy, evaluate_diversity_multimodal
from .evaluator import MultimodalEvaluator
from .fidelity import FidelityEvaluator, compute_jsd, evaluate_fidelity_multimodal
from .privacy import PrivacyEvaluator, evaluate_privacy_multimodal
from .utility import UtilityEvaluator, evaluate_utility_multimodal

__all__ = [
    "UtilityEvaluator",
    "FidelityEvaluator",
    "PrivacyEvaluator",
    "DiversityEvaluator",
    "MultimodalEvaluator",
    "auto_detect_columns",
    "compute_jsd",
    "compute_joint_entropy",
    "evaluate_fidelity_multimodal",
    "evaluate_utility_multimodal",
    "evaluate_diversity_multimodal",
    "evaluate_privacy_multimodal",
]

