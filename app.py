import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Dermatology Disease Classification",
    layout="centered"
)

st.title("🩺 Dermatology Image Classification")
st.write("Upload a skin image to predict the disease class.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "efficientnet_dermatology_final.keras"
    )

model = load_model()

# ---------------- LOAD CLASS NAMES ----------------
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ---------------- IMAGE PREPROCESS ----------------
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            img_tensor = preprocess_image(image)
            preds = model.predict(img_tensor)[0]

            top_idx = np.argmax(preds)
            confidence = preds[top_idx] * 100

            st.success(f"Prediction: **{class_names[top_idx]}**")
            st.write(f"Confidence: **{confidence:.2f}%**")

            # Optional: show top-3
            st.subheader("Top Predictions")
            top3 = np.argsort(preds)[-3:][::-1]
            for i in top3:
                st.write(f"{class_names[i]} : {preds[i]*100:.2f}%")
