# Model Card — Vehicle Damage Detection

## Intended use
Portfolio/demo computer-vision inference for classifying visible vehicle damage into seven categories. The service is not an insurance decision engine, repair estimator, or safety-critical diagnostic system.

The model always returns one of its seven trained categories. It has no trained no-damage/unknown output and no validated out-of-distribution detector. The API's review flag is a confidence heuristic and must not be interpreted as reliable unknown-image rejection.

## Model
- Architecture: MobileNetV2 transfer learning
- Input: RGB 224×224 image, scaled to [0, 1]
- Output: seven-class softmax
- Served artifact: `model/car_damage_model.keras`
- Model version: configurable through `MODEL_VERSION` (default `1.0.0`)

## Data and evaluation
The documented clean run uses the deduplicated dataset with the `unknown` class removed, followed by a stratified 70/15/15 split. The exact split files are persisted under `artifacts/splits/`. The test split is not used for training or fine-tuning.

## Reported test performance
- Accuracy: 78.32%
- Macro F1: 0.7734
- Weighted F1: 0.7834
- ECE: 0.0679
- Brier score: 0.3293

These are the project’s verified recorded results for the documented run. They are not a guarantee of performance on new real-world images.

## Known limitations
- Dataset size is modest and the test set has 143 images.
- The test images come from the same small source dataset and do not establish generalization across capture devices, regions, vehicle types, or real inspection workflows.
- Images outside the seven labels can be assigned a damage label with high confidence; the review heuristic does not solve this reliably.
- `bumper_scratch` is the weakest documented class (F1 0.571 in the frozen-model test report).
- The dataset is not equivalent to an operational insurance/repair inspection corpus.
- The saved random stratified split is useful for reproducing the current baseline, but future evaluation should use grouped splits by vehicle/source/session to reduce dependence between related images.
- Grad-CAM is an explanation aid, not proof that the highlighted region is the true damage cause.
- Severity and repair-cost outputs are heuristics/illustrative values and must not be presented as engineering measurements or quotes.

## Uncertainty handling
The API reports top probability, second-best probability, margin, normalized entropy, and a `review_recommended` flag. The thresholds are configurable environment variables and are decision-support rules, not calibrated correctness probabilities.

## Reproducibility
Training and evaluation scripts persist their split/configuration artifacts. The repository keeps model hashes in `model/model_manifest.json`. The legacy `.h5` artifact is retained for provenance and is not served.
