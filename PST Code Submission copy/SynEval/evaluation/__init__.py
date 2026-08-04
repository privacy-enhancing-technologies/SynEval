"""
Evaluation module for SynEval framework.

This module contains all evaluation components:
- UtilityEvaluator: Evaluates utility metrics
- FidelityEvaluator: Evaluates fidelity metrics
- PrivacyEvaluator: Evaluates privacy metrics
- DiversityEvaluator: Evaluates diversity metrics
"""

def __getattr__(name):
    """Lazy import to avoid loading all dependencies at once."""
    if name == "DiversityEvaluator":
        from .diversity import DiversityEvaluator
        return DiversityEvaluator
    elif name == "FidelityEvaluator":
        from .fidelity import FidelityEvaluator
        return FidelityEvaluator
    elif name == "PrivacyEvaluator":
        from .privacy import PrivacyEvaluator
        return PrivacyEvaluator
    elif name == "UtilityEvaluator":
        from .utility import UtilityEvaluator
        return UtilityEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "UtilityEvaluator",
    "FidelityEvaluator",
    "PrivacyEvaluator",
    "DiversityEvaluator",
]

