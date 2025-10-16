# Plan: Resolve `torch` Import Failure and Documentation Drift

## Background
- Following the README instructions currently clones the obsolete `SCU-TrustworthyAI/SynEval` repository instead of the active `privacy-enhancing-technologies/SynEval`.
- Executing `python run.py ...` or the demo notebook’s `!run.py` cell in a fresh environment triggers `ImportError: No module named 'torch'`, indicating the framework hard-depends on PyTorch even for CPU-only use cases.

## Planned Work
1. **Reproduce & Diagnose**
   - Set up a clean virtual environment that follows the existing README steps to confirm both the outdated clone instructions and the `torch` import failure.
   - Document the exact stack trace and identify which modules require PyTorch at import time.
2. **Improve Dependency Handling**
   - Refactor `run.py` and the evaluator modules (`fidelity.py`, `diversity.py`, `privacy.py`, `utility.py`, plus any helpers) to guard their `torch` usage behind lazy imports or graceful fallbacks.
   - Ensure CPU-only paths remain functional when PyTorch is absent, and emit actionable messaging when a specific metric truly requires it.
3. **Update Documentation & Notebook**
   - Correct the README cloning instructions to point to `privacy-enhancing-technologies/SynEval`.
   - Revise setup guidance to clarify when PyTorch is optional vs. required and how to install the appropriate wheel.
   - Adjust `SynEval_Demo.ipynb` cells so they invoke `python run.py ...` (instead of `!run.py`) and align with the new dependency guidance.
4. **Validation**
   - Run the normal test suite (`pytest`) plus a targeted invocation of `run.py` in an environment without PyTorch to verify the graceful degradation.
   - Where feasible, add regression coverage that simulates missing PyTorch (e.g., patching `sys.modules`).
5. **Contribution Workflow**
   - Create a feature branch off `main`, follow PEP 8 style, and run `black`/`flake8` before committing as directed in `CONTRIBUTING.md`.
   - Prepare a PR with a clear summary, reproduction notes, test evidence, and any follow-up actions.

## Deliverables
- Code changes implementing optional PyTorch handling.
- Updated README and notebook instructions that reflect the correct repository and installation process.
- Test results demonstrating both the PyTorch-free fallback and the standard test suite passing.

## Implementation Summary
- Wrapped `torch` and Flair imports in `try/except` blocks across `run.py`, `utility.py`, `diversity.py`, `fidelity.py`, and `privacy.py`, exposing `TORCH_AVAILABLE` flags so CPU-only environments do not fail at import time.
- Added device selection fallbacks and ImportError-driven skips in `run.py` so individual evaluation dimensions are bypassed with clear messaging when GPU-only dependencies are missing.
- Preserved existing GPU-accelerated code paths when `torch` is present by routing through the new availability checks instead of removing CUDA logic.
