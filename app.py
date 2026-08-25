"""
app.py
-------
Streamlit web app: user uploads any image, the app resizes it to 32x32,
feeds it into the trained CIFAR-10 CNN, and shows the predicted class
name (airplane, automobile, bird, cat, deer, dog, frog, horse, ship,
truck) with confidence scores for all 10 classes.

Run locally:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud:
    1. Push this folder (app.py, train_model.py, cifar10_model.h5,
       requirements.txt) to a public GitHub repo.
    2. Go to https://share.streamlit.io -> "New app".
    3. Pick your repo, branch, and set Main file path = app.py.
    4. Click Deploy. You'll get a shareable link like:
       https://<your-app-name>.streamlit.app
"""

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_model.h5")


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((32, 32))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


st.title("🖼️ CIFAR-10 Image Classifier")
st.write(
    "Upload any image and the model will guess which of the 10 CIFAR-10 "
    "categories it belongs to: **airplane, automobile, bird, cat, deer, "
    "dog, frog, horse, ship, truck**."
)
st.caption(
    "Note: the model only knows these 10 categories, so images outside "
    "them (e.g. a laptop or a person) will still be forced into the "
    "closest matching class."
)

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Predicting..."):
        input_arr = preprocess(image)
        predictions = model.predict(input_arr)[0]

    top_idx = int(np.argmax(predictions))
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(predictions[top_idx]) * 100

    st.success(f"**Prediction: {top_class.upper()}**  ({top_conf:.1f}% confidence)")

    st.subheader("Confidence for all classes")
    sorted_pairs = sorted(zip(CLASS_NAMES, predictions), key=lambda p: p[1], reverse=True)
    for name, score in sorted_pairs:
        st.write(f"{name}")
        st.progress(float(score))

st.divider()
st.caption("Built with TensorFlow/Keras + Streamlit · CIFAR-10 CNN classifier")
