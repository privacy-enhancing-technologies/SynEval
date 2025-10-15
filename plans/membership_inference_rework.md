# Plan: Rework “Membership Inference” Metric

1. **Define correct threat model**
   - Document the discrepancy between SynEval’s current classifier-based distinguishability metric and true model-centric membership inference.
   - Propose switching terminology (e.g., “distribution distinguishability”) or adding a separate, model-based privacy test.
2. **Design improved evaluation**
   - Add an optional workflow that trains a downstream model on real data, then applies a classical membership inference attack (confidence-based or loss-based) against held-out samples.
   - Retain the existing distinguishability score but clearly label it, emit guidance when high scores are actually evidence of safer (more private) synthetic data.
3. **Implementation outline**
   - Refactor `_evaluate_membership_inference` to separate feature generation from attack logic.
   - Introduce a new module (e.g., `privacy_membership.py`) with configurable attack strategies (shadow models, threshold attacks) and plug it into the CLI via `--privacy-metrics`.
   - Provide configuration flags/metadata entries to point at a target model and evaluation datasets.
4. **Validation plan**
   - Create synthetic test cases: (a) identical real/synthetic datasets (should yield high risk) and (b) heavily noised synthetic data (should yield low risk in true MIA, high distinguishability).
   - Update unit tests to assert naming clarity and correctness of both metrics.
5. **Documentation & migration**
   - Update README/privacy docs to clarify the distinction, migration path, and expected outputs.
   - Deprecate the old metric name in a subsequent release after communicating the change.
