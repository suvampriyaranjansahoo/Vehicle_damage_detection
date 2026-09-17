"""Batch CLI: python scripts/batch_predict.py --dir photos --output artifacts/batch_report.pdf."""

import argparse
import base64
import logging
from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.config import CLASS_NAMES_PATH
from app.inference import predict
from app.labels import load_class_names

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--output", default="artifacts/batch_report.pdf")
    args = parser.parse_args()

    names = load_class_names(CLASS_NAMES_PATH)
    results = []

    for file_path in sorted(Path(args.dir).iterdir()):
        if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue

        try:
            results.append((file_path, predict(Image.open(file_path), names)))
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Prediction failed for %s", file_path)
            results.append((file_path, {"error": str(error)}))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(str(output), pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Vehicle Damage Batch Report", styles["Title"]), Spacer(1, 12)]

    for file_path, result in results:
        story.append(Paragraph(f"<b>{file_path.name}</b>", styles["Heading2"]))

        if "error" in result:
            story.append(Paragraph(result["error"], styles["BodyText"]))
            continue

        story.append(
            Paragraph(
                (
                    f"Damage: {result['predicted_class']} | "
                    f"Confidence: {result['confidence']:.2%} | "
                    f"Severity: {result['severity']['level']} "
                    f"({result['severity']['score']})"
                ),
                styles["BodyText"],
            )
        )
        story.append(
            Paragraph(
                (
                    "Illustrative repair range: "
                    f"₹{result['repair_cost']['min']:,}–"
                    f"₹{result['repair_cost']['max']:,}"
                ),
                styles["BodyText"],
            )
        )

        try:
            image_data = base64.b64decode(result["gradcam_overlay_base64"])
            temporary_image = Path("/tmp/gradcam.jpg")
            temporary_image.write_bytes(image_data)
            story.append(RLImage(str(temporary_image), width=280, height=200))
        except Exception:  # noqa: BLE001
            LOGGER.warning("Could not embed Grad-CAM overlay for %s", file_path.name)

        story.append(Spacer(1, 12))

    document.build(story)
    print(output)


if __name__ == "__main__":
    main()
