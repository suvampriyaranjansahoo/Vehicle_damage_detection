import base64
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def test_health_check():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_invalid_file_type_rejected():
    r = client.post('/predict', files={'file': ('x.txt', b'not an image', 'text/plain')})
    assert r.status_code == 415


def test_valid_image_prediction(monkeypatch):
    fake = {
        'predicted_class': 'bumper_dent', 'confidence': 0.9,
        'probabilities': {'bumper_dent': 0.9},
        'severity': {'score': 40, 'level': 'moderate', 'activation_area': .1, 'activation_intensity': .2},
        'repair_cost': {'currency': 'INR', 'illustrative': True, 'min': 3000, 'max': 10000},
        'gradcam_overlay_base64': base64.b64encode(b'fake').decode(),
    }
    monkeypatch.setattr('app.main.predict', lambda image, classes: fake)
    monkeypatch.setattr('app.main.log_prediction', lambda result: None)
    buf = BytesIO(); Image.new('RGB', (64, 64), 'white').save(buf, format='JPEG')
    r = client.post('/predict', files={'file': ('car.jpg', buf.getvalue(), 'image/jpeg')})
    assert r.status_code == 200
    assert r.json()['predicted_class'] == 'bumper_dent'

def test_class_mapping_is_seven_unique_labels():
    from app.config import CLASS_NAMES_PATH
    from app.labels import load_class_names
    names = load_class_names(CLASS_NAMES_PATH)
    assert len(names) == 7
    assert len(set(names)) == 7
