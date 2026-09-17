import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    labels=json.loads(Path('class_names.json').read_text())
    model_path=Path('model/car_damage_model.keras')
    if not model_path.exists(): raise FileNotFoundError(model_path)
    import tensorflow as tf
    model=tf.keras.models.load_model(model_path)
    x=np.zeros((1,224,224,3),dtype=np.float32); y=model.predict(x,verbose=0)
    assert y.shape==(1,len(labels)) and np.isfinite(y).all()
    Image.new('RGB',(224,224),'white').save('/tmp/smoke.jpg')
    print('Smoke test passed:', y.shape)
if __name__=='__main__':main()
