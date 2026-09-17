"""Benchmark MobileNetV2 vs EfficientNetB0 when dataset images are available."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build(name,n):
    base=(MobileNetV2 if name=="MobileNetV2" else EfficientNetB0)(input_shape=(224,224,3),include_top=False,weights="imagenet"); base.trainable=False
    m=tf.keras.Sequential([base,layers.GlobalAveragePooling2D(),layers.Dense(128,activation="relu"),layers.Dropout(.3),layers.Dense(n,activation="softmax")]); m.compile("adam","categorical_crossentropy",metrics=["accuracy"]); return m

def main():
    p=argparse.ArgumentParser();p.add_argument("--csv",default="data/data.csv");p.add_argument("--image-root",default=".");p.add_argument("--epochs",type=int,default=2);a=p.parse_args()
    df=pd.read_csv(a.csv);df["path"]=df.image.map(lambda x:str(Path(a.image_root)/x))
    if df.path.map(lambda x:Path(x).is_file()).sum()!=len(df):raise FileNotFoundError("Referenced images are missing; benchmark cannot run.")
    tr,te=train_test_split(df,test_size=.2,stratify=df.classes,random_state=42); g=ImageDataGenerator(rescale=1./255);kw={"x_col":"path","y_col":"classes","target_size":(224,224),"batch_size":32,"class_mode":"categorical"}
    tg=g.flow_from_dataframe(tr,shuffle=True,**kw);eg=g.flow_from_dataframe(te,shuffle=False,**kw);rows=[]
    for name in ["MobileNetV2","EfficientNetB0"]:
      m=build(name,tg.num_classes);m.fit(tg,epochs=a.epochs,verbose=0); t=time.perf_counter();probs=m.predict(eg,verbose=0);lat=(time.perf_counter()-t)/len(te)*1000;y=np.argmax(probs,1);rows.append({"backbone":name,"accuracy":accuracy_score(eg.classes,y),"macro_f1":f1_score(eg.classes,y,average="macro"),"inference_ms_per_image":lat,"model_size_mb":Path(f"/tmp/{name}.keras").stat().st_size/1e6 if Path(f"/tmp/{name}.keras").exists() else None});m.save(f"/tmp/{name}.keras")
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/backbone_comparison.json","w") as f:
        json.dump(rows,f,indent=2)
    print(json.dumps(rows,indent=2))
if __name__=="__main__":main()
