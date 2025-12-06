import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('car_damage_model.h5')

# Make sure this list matches the number and order of classes used during training
class_names = [
    'other',
    'bumper_dent'
    'bumper_scratch',
    'door_dent',
    'door_scratch',
    'glass_shatter',
    'head_lamp',
    'tail_lamp',
    'unknown'
    
]

st.title("🚗 Car Damage Detection App")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)

        st.write("📊 Raw predictions:", predictions)
        st.write("🔢 Predicted index:", predicted_index)

        if predicted_index < len(class_names):
            predicted_class = class_names[predicted_index]
            st.success(f"✅ Predicted Damage Type: **{predicted_class}**")
        else:
            st.error(f"❌ Error: Predicted index {predicted_index} is out of range.")
