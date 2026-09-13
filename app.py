```python
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Streamlit page configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)


# Load trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_model.h5")


# Preprocess image
def preprocess(image: Image.Image) -> np.ndarray:
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# App title
st.title("🖼️ CIFAR-10 Image Classifier")

st.write(
    "Upload any image and the model will guess which of the 10 CIFAR-10 "
    "categories it belongs to: **airplane, automobile, bird, cat, deer, "
    "dog, frog, horse, ship, truck**."
)

st.caption(
    "Note: the model only knows these 10 categories, so images outside "
    "them (for example, a laptop or a person) will still be forced into "
    "the closest matching class."
)


# Load model
try:
    model = load_model()

except Exception as e:
    st.error("Unable to load the trained model.")
    st.error(f"Error: {e}")
    st.stop()


# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read image
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


    # Show uploaded image
    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )


    # Make prediction
    with st.spinner("Predicting..."):

        # Resize image to CIFAR-10 input size
        resized = image.resize((32, 32))

        # Preprocess image
        input_arr = preprocess(resized)

        # Model prediction
        predictions = model.predict(input_arr, verbose=0)[0]


    # Get highest probability class
    top_idx = int(np.argmax(predictions))

    top_class = CLASS_NAMES[top_idx]

    top_conf = float(predictions[top_idx]) * 100


    # Display prediction
    st.success(
        f"**Prediction: {top_class.upper()}** "
        f"({top_conf:.1f}% confidence)"
    )


    # Display confidence scores
    st.subheader("Confidence for all classes")

    sorted_pairs = sorted(
        zip(CLASS_NAMES, predictions),
        key=lambda p: p[1],
        reverse=True
    )


    for name, score in sorted_pairs:

        st.write(f"{name}")

        st.progress(float(score))


# Footer
st.divider()

st.caption(
    "Built with TensorFlow/Keras + Streamlit · CIFAR-10 CNN classifier"
)
```
