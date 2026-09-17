"""Batch CLI: python scripts/batch_predict.py --dir photos --output artifacts/batch_report.pdf"""
import argparse
import base64
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.config import CLASS_NAMES_PATH
from app.inference import predict
from app.labels import load_class_names


def main():
 p=argparse.ArgumentParser();p.add_argument('--dir',required=True);p.add_argument('--output',default='artifacts/batch_report.pdf');a=p.parse_args();
 names=load_class_names(CLASS_NAMES_PATH); results=[]
 for f in sorted(Path(a.dir).iterdir()):
  if f.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}: continue
  try:
   results.append((f,predict(Image.open(f),names)))
  except Exception as e:
   LOGGER.exception("Prediction failed for %s", f)
   results.append((f,{'error':str(e)}))
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);doc=SimpleDocTemplate(str(out),pagesize=A4);styles=getSampleStyleSheet();story=[Paragraph('Vehicle Damage Batch Report',styles['Title']),Spacer(1,12)]
 for f,r in results:
  story.append(Paragraph(f'<b>{f.name}</b>',styles['Heading2']))
  if 'error' in r: story.append(Paragraph(r['error'],styles['BodyText']));continue
  story.append(Paragraph(f"Damage: {r['predicted_class']} | Confidence: {r['confidence']:.2%} | Severity: {r['severity']['level']} ({r['severity']['score']})",styles['BodyText']))
  story.append(Paragraph(f"Illustrative repair range: ₹{r['repair_cost']['min']:,}–₹{r['repair_cost']['max']:,}",styles['BodyText']))
  try:
   img=base64.b64decode(r['gradcam_overlay_base64']);tmp=Path('/tmp/gradcam.jpg');tmp.write_bytes(img);story.append(RLImage(str(tmp),width=280,height=200))
  except Exception:  # noqa: BLE001 -- best-effort report embed, must not abort the whole batch
   LOGGER.warning("Could not embed Grad-CAM overlay for %s", f.name)
  story.append(Spacer(1,12))
 doc.build(story)
 print(out)
if __name__=='__main__':main()
