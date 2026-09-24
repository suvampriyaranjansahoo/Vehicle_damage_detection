# Portfolio readiness roadmap

This roadmap defines what would support a 9.5/10 portfolio presentation. That rating is subjective; model performance and product readiness must be supported by evidence rather than claimed in advance.

## Current baseline

- Seven damage labels; the model always chooses one of them.
- Frozen MobileNetV2: 78.32% accuracy and 0.7734 macro-F1 on the saved 143-image test split.
- The weakest reported class is `bumper_scratch` (F1 0.571 on 12 test examples).
- Training used 665 images. The small test set makes the measured score uncertain and limits claims about real-world use.
- Severity, cost, Grad-CAM, and confidence-based review are demo aids, not validated inspection tools.

## Milestone 1 — Build a stronger evaluation set

1. Gather additional labeled photos across every supported category, with varied vehicles, viewpoints, lighting, distance, and backgrounds. Include images with no visible damage and damage outside the seven labels.
2. Obtain clear permission for images and remove identifying details where practical. Record source and label provenance.
3. Have labels checked by a second reviewer. Resolve disagreements before training.
4. Group related photos by vehicle, source, or capture session before splitting. Keep a final test group isolated from all training and tuning.
5. Publish sample counts by class and capture condition. As a practical first target, seek at least 200 diverse examples per supported class and a similarly varied no-damage/out-of-scope set; more may be needed for stable estimates.

## Milestone 2 — Compare models without test-set tuning

1. Freeze the grouped train/validation/test manifests and keep the existing test set as a historical baseline.
2. Compare the current MobileNetV2 model with a fair fine-tuning run and at least one alternative backbone. Select settings using validation results only.
3. Track accuracy, macro-F1, per-class precision/recall, calibration, model size, and inference latency.
4. Evaluate each finalist once on the untouched test set. Report confidence intervals or otherwise show the uncertainty caused by sample size.
5. Promote a new served model only when it improves the chosen metrics on the held-out data and has a reproducible config and artifact hash. Do not claim 90% unless a sufficiently broad independent test set supports it.

## Milestone 3 — Handle uncertain and out-of-scope photos

1. Add a representative no-damage and out-of-scope dataset.
2. Choose an abstention/review policy on validation data, then measure false acceptance and false rejection on the untouched test set.
3. Ensure the interface explains when the image is uncertain and makes clear that the confidence score is not a correctness guarantee.
4. Keep severity and cost labeled as illustrative until compared with expert assessments and real repair quotes.

## Milestone 4 — Make the demo and repository review-ready

- The Streamlit Analyze screen presents the result, alternative labels, review status, Grad-CAM, model version, and clear limitations.
- The monitoring view is described as usage monitoring, not accuracy monitoring, unless ground-truth labels are collected.
- The README gives a clean setup path, a reproducible evaluation command, concise results, and known limitations.
- CI should verify lint, API contracts, model loading, and a small smoke prediction. Keep checks fast and deterministic.
- Include a short demo recording and one-page model card once the upgraded evaluation is complete.

## Acceptance evidence

Call the project portfolio-ready at this level when a clean setup works, the demo communicates its limits, model artifacts are traceable, the independent grouped evaluation is reproducible, and performance is reported per class with uncertainty. A 9.5/10 judgment still depends on presentation quality and reviewer expectations; no single accuracy threshold guarantees it.
