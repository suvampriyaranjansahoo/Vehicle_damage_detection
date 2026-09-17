"""Fine-tune the top N layers of MobileNetV2 after a frozen baseline."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    p=argparse.ArgumentParser(); p.add_argument("--csv",default="data/data.csv"); p.add_argument("--image-root",default="."); p.add_argument("--top-n",type=int,default=30); p.add_argument("--epochs",type=int,default=5); p.add_argument("--model",default="artifacts/mobile_netv2_frozen.keras"); args=p.parse_args()
    df=pd.read_csv(args.csv); df["path"]=df.image.map(lambda x:str(Path(args.image_root)/x))
    if df.path.map(lambda x:Path(x).is_file()).sum()!=len(df): raise FileNotFoundError("Referenced images are missing; fine-tuning cannot run.")
    tr,va=train_test_split(df,test_size=.2,stratify=df.classes,random_state=42)
    gen=ImageDataGenerator(rescale=1./255,rotation_range=15,zoom_range=.1,horizontal_flip=True); plain=ImageDataGenerator(rescale=1./255)
    kw={"x_col":"path","y_col":"classes","target_size":(224,224),"batch_size":32,"class_mode":"categorical"}
    tg=gen.flow_from_dataframe(tr,shuffle=True,seed=42,**kw); vg=plain.flow_from_dataframe(va,shuffle=False,**kw)
    model=tf.keras.models.load_model(args.model); base=model.layers[0]; base.trainable=True
    for layer in base.layers[:-args.top_n]: layer.trainable=False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss="categorical_crossentropy",metrics=["accuracy"])
    w=compute_class_weight("balanced",classes=np.arange(tg.num_classes),y=tg.classes)
    model.fit(tg,validation_data=vg,epochs=args.epochs,class_weight=dict(enumerate(w)))
    Path("artifacts").mkdir(exist_ok=True); model.save("artifacts/mobile_netv2_finetuned.keras")
    print("Saved artifacts/mobile_netv2_finetuned.keras")
if __name__=="__main__":main()
