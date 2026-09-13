import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_model.h5")

def preprocess(image: Image.Image) -> np.ndarray:
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

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.error(f"Error: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
    except UnidentifiedImageError:
        st.error(
            "That file doesn't look like a valid image. "
            "Please upload a JPG or PNG."
        )
        st.stop()
    except Exception as e:
        st.error(f"Couldn't read that image: {e}")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Predicting..."):
        resized = image.resize((32, 32))
        input_arr = preprocess(resized)
        predictions = model.predict(input_arr, verbose=0)[0]

    top_idx = int(np.argmax(predictions))
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(predictions[top_idx]) * 100

    st.success(
        f"**Prediction: {top_class.upper()}** "
        f"({top_conf:.1f}% confidence)"
    )

    st.subheader("Confidence for all classes")

    sorted_pairs = sorted(
        zip(CLASS_NAMES, predictions),
        key=lambda p: p[1],
        reverse=True
    )

    for name, score in sorted_pairs:
        st.write(f"{name}")
        st.progress(float(score))

st.divider()

st.caption(
    "Built with TensorFlow/Keras + Streamlit · CIFAR-10 CNN classifier"
)
