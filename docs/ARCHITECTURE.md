# System Architecture

```text
                         ┌──────────────────────┐
                         │      Streamlit UI    │
                         │ Analyze + Monitoring │
                         └──────────┬───────────┘
                                    │ HTTP
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │ /health /ready      │
                         │ /predict /monitoring│
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
             ┌────────────┐ ┌─────────────┐ ┌──────────────┐
             │ Keras      │ │ Grad-CAM    │ │ JSONL        │
             │ MobileNetV2│ │ explanation │ │ monitoring   │
             └────────────┘ └─────────────┘ └──────────────┘
                    │               │                │
                    └───────────────┴────────────────┘
                                    ▼
                         Prediction + uncertainty
                         + heuristic severity/cost
```

## Request flow
1. API validates extension and bounded upload size.
2. Pillow verifies and decodes the image.
3. The model loader resolves the local artifact (or configured remote fallback) and validates its output contract against `class_names.json`.
4. Inference returns class probabilities and top-3 predictions.
5. Uncertainty rules flag low-confidence cases for review.
6. Grad-CAM produces a visual explanation.
7. Heuristic severity and illustrative cost are returned with explicit caveats.
8. Prediction metadata and latency are appended to JSONL monitoring.

## Deployment
The API and UI are separate containers. Docker Compose uses the API readiness probe before starting the UI dependency. Render can use `/ready` as the service health check.
