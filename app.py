import os
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

# ---------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CLASS_EMOJIS = {
    "airplane": "✈️",
    "automobile": "🚗",
    "bird": "🐦",
    "cat": "🐱",
    "deer": "🦌",
    "dog": "🐕",
    "frog": "🐸",
    "horse": "🐴",
    "ship": "🚢",
    "truck": "🚛",
}

CLASS_GRADIENTS = {
    "airplane": ("#60a5fa", "#06b6d4"),
    "automobile": ("#f87171", "#f97316"),
    "bird": ("#facc15", "#f59e0b"),
    "cat": ("#c084fc", "#ec4899"),
    "deer": ("#fbbf24", "#ca8a04"),
    "dog": ("#fb923c", "#ef4444"),
    "frog": ("#4ade80", "#059669"),
    "horse": ("#fb7185", "#db2777"),
    "ship": ("#818cf8", "#2563eb"),
    "truck": ("#94a3b8", "#4b5563"),
}

MODEL_PATH = "cifar10_model.h5"

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Custom CSS - designed to closely match the Arena version
# ---------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 85% 5%, rgba(168,85,247,.14), transparent 28%),
        radial-gradient(circle at 10% 80%, rgba(6,182,212,.10), transparent 25%),
        linear-gradient(135deg, #020617 0%, #170b2d 48%, #020617 100%);
    color: white;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: .45;
    background-image: radial-gradient(circle at 1px 1px,
        rgba(255,255,255,.035) 1px, transparent 0);
    background-size: 40px 40px;
}

.block-container {
    max-width: 1180px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: rgba(2,6,23,.65);
}

.topbar {
    border-bottom: 1px solid rgba(255,255,255,.07);
    padding: 12px 0 18px;
    margin-bottom: 42px;
}

.brand-row {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:20px;
}

.brand-left {
    display:flex;
    align-items:center;
    gap:12px;
}

.logo {
    width:42px;
    height:42px;
    border-radius:13px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:21px;
    background:linear-gradient(135deg,#8b5cf6,#06b6d4);
    box-shadow:0 10px 30px rgba(139,92,246,.25);
}

.brand-title {
    color:#fff;
    font-size:18px;
    font-weight:800;
    line-height:1.2;
}

.brand-subtitle {
    color:rgba(255,255,255,.38);
    font-size:11px;
    margin-top:3px;
}

.status {
    padding:7px 12px;
    border-radius:999px;
    color:#34d399;
    background:rgba(16,185,129,.10);
    border:1px solid rgba(16,185,129,.20);
    font-size:11px;
    font-weight:600;
}

.status-dot {
    display:inline-block;
    width:7px;
    height:7px;
    border-radius:50%;
    background:#34d399;
    margin-right:7px;
}

.hero {
    text-align:center;
    margin-bottom:34px;
}

.badge {
    display:inline-block;
    padding:7px 14px;
    border-radius:999px;
    color:rgba(255,255,255,.62);
    background:rgba(255,255,255,.045);
    border:1px solid rgba(255,255,255,.09);
    font-size:11px;
    font-weight:600;
    margin-bottom:20px;
}

.hero h1 {
    color:#fff;
    font-size:clamp(38px,6vw,64px);
    line-height:1.05;
    letter-spacing:-2.5px;
    margin:0 0 18px;
    font-weight:800;
}

.gradient-text {
    background:linear-gradient(90deg,#a78bfa,#22d3ee,#34d399);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero p {
    max-width:720px;
    margin:auto;
    color:rgba(255,255,255,.48);
    font-size:16px;
    line-height:1.7;
}

.feature {
    min-height:90px;
    display:flex;
    align-items:center;
    gap:14px;
    padding:17px;
    border-radius:15px;
    background:rgba(255,255,255,.035);
    border:1px solid rgba(255,255,255,.065);
}

.feature-icon {
    font-size:23px;
}

.feature-title {
    color:rgba(255,255,255,.9);
    font-size:13px;
    font-weight:700;
}

.feature-desc {
    color:rgba(255,255,255,.38);
    font-size:11px;
    margin-top:4px;
}

.section-title {
    color:rgba(255,255,255,.78);
    font-size:14px;
    font-weight:700;
    margin:8px 0 13px;
}

.upload-card {
    border:2px dashed rgba(255,255,255,.11);
    border-radius:20px;
    padding:38px 25px;
    text-align:center;
    background:rgba(255,255,255,.025);
}

.upload-icon {
    font-size:42px;
    margin-bottom:10px;
}

.upload-title {
    color:rgba(255,255,255,.88);
    font-size:16px;
    font-weight:700;
}

.upload-text {
    color:rgba(255,255,255,.38);
    font-size:12px;
    margin-top:7px;
}

.image-card {
    border:1px solid rgba(255,255,255,.09);
    border-radius:20px;
    padding:14px;
    background:rgba(255,255,255,.025);
}

.result-card {
    border-radius:20px;
    overflow:hidden;
    border:1px solid rgba(255,255,255,.10);
    margin-top:18px;
}

.result-inner {
    padding:25px;
    color:white;
}

.result-label {
    color:rgba(255,255,255,.78);
    font-size:12px;
    font-weight:600;
    margin-bottom:7px;
}

.prediction-row {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:20px;
}

.prediction-main {
    display:flex;
    align-items:center;
    gap:14px;
}

.prediction-emoji {
    font-size:43px;
}

.prediction-name {
    font-size:31px;
    font-weight:800;
    text-transform:capitalize;
}

.prediction-confidence {
    color:rgba(255,255,255,.78);
    font-size:12px;
    margin-top:3px;
}

.confidence-circle {
    width:78px;
    height:78px;
    border:4px solid rgba(255,255,255,.28);
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:18px;
    font-weight:800;
}

.confidence-box {
    border:1px solid rgba(255,255,255,.09);
    background:rgba(255,255,255,.025);
    border-radius:18px;
    padding:17px;
    margin-top:14px;
}

.confidence-heading {
    color:rgba(255,255,255,.58);
    font-size:12px;
    font-weight:600;
    margin-bottom:13px;
}

.class-row {
    margin-bottom:12px;
}

.class-line {
    display:flex;
    align-items:center;
    justify-content:space-between;
    margin-bottom:5px;
}

.class-name {
    color:rgba(255,255,255,.70);
    font-size:12px;
}

.class-score {
    color:rgba(255,255,255,.48);
    font-size:12px;
    font-family:monospace;
}

.class-score.top {
    color:white;
    font-weight:800;
}

.bar-bg {
    height:8px;
    border-radius:99px;
    background:rgba(255,255,255,.055);
    overflow:hidden;
}

.bar {
    height:100%;
    border-radius:99px;
}

.arch-card {
    margin-top:45px;
    padding:24px;
    border-radius:18px;
    border:1px solid rgba(255,255,255,.09);
    background:linear-gradient(135deg,rgba(139,92,246,.06),rgba(6,182,212,.05));
}

.arch-text {
    color:rgba(255,255,255,.45);
    font-size:12px;
    line-height:1.7;
}

.stat {
    text-align:center;
    padding:13px;
    border-radius:13px;
    background:rgba(255,255,255,.045);
    border:1px solid rgba(255,255,255,.075);
}

.stat-value {
    color:#a78bfa;
    font-size:18px;
    font-weight:800;
}

.stat-label {
    color:rgba(255,255,255,.35);
    font-size:10px;
    margin-top:2px;
}

.how-card {
    margin-top:20px;
    padding:24px;
    border-radius:18px;
    border:1px solid rgba(255,255,255,.08);
    background:rgba(255,255,255,.025);
    text-align:center;
}

.step-icon {
    width:48px;
    height:48px;
    margin:0 auto 10px;
    border-radius:13px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:24px;
    background:linear-gradient(135deg,rgba(139,92,246,.20),rgba(6,182,212,.20));
}

.step-number {
    width:24px;
    height:24px;
    margin:0 auto 8px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    background:rgba(139,92,246,.20);
    color:#a78bfa;
    font-size:10px;
    font-weight:800;
}

.step-title {
    color:rgba(255,255,255,.80);
    font-size:12px;
    font-weight:700;
}

.step-desc {
    color:rgba(255,255,255,.35);
    font-size:10px;
    line-height:1.5;
    margin-top:4px;
}

.classes-title {
    color:rgba(255,255,255,.55);
    text-align:center;
    font-size:12px;
    font-weight:600;
    margin:42px 0 15px;
}

.class-chip {
    text-align:center;
    padding:12px 5px;
    border-radius:12px;
    background:rgba(255,255,255,.025);
    border:1px solid rgba(255,255,255,.06);
}

.class-chip.selected {
    background:rgba(255,255,255,.09);
    border-color:rgba(139,92,246,.50);
    box-shadow:0 10px 30px rgba(139,92,246,.10);
}

.class-chip-emoji {
    font-size:25px;
}

.class-chip-name {
    color:rgba(255,255,255,.48);
    font-size:9px;
    text-transform:capitalize;
    margin-top:4px;
}

.share-card {
    margin-top:48px;
    padding:32px;
    border-radius:20px;
    text-align:center;
    border:1px solid rgba(255,255,255,.09);
    background:linear-gradient(90deg,rgba(139,92,246,.10),rgba(6,182,212,.08));
}

.footer {
    margin-top:45px;
    padding:25px 0;
    border-top:1px solid rgba(255,255,255,.07);
    color:rgba(255,255,255,.30);
    text-align:center;
    font-size:10px;
}

div[data-testid="stFileUploader"] {
    background:transparent;
}

div[data-testid="stFileUploader"] section {
    border:none !important;
    padding:0 !important;
    background:transparent !important;
}

.stButton > button {
    width:100%;
    border-radius:12px;
    border:1px solid rgba(255,255,255,.15);
    background:rgba(255,255,255,.07);
    color:white;
    font-weight:600;
}

.stButton > button:hover {
    border-color:rgba(139,92,246,.55);
    background:rgba(139,92,246,.15);
}

[data-testid="stMetric"] {
    background:rgba(255,255,255,.03);
    border:1px solid rgba(255,255,255,.07);
    padding:12px;
    border-radius:13px;
}

[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color:white !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} was not found. Put the trained model beside app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image):
    image = image.resize((32, 32))
    arr = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def gradient_css(class_name):
    c1, c2 = CLASS_GRADIENTS[class_name]
    return f"linear-gradient(90deg,{c1},{c2})"


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
<div class="topbar">
  <div class="brand-row">
    <div class="brand-left">
      <div class="logo">✨</div>
      <div>
        <div class="brand-title">CIFAR-10 Classifier</div>
        <div class="brand-subtitle">Deep Learning Image Recognition</div>
      </div>
    </div>
    <div class="status"><span class="status-dot"></span>Model Active</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="badge">🧠 &nbsp; Convolutional Neural Network</div>
  <h1>Image Classification<br>
    <span class="gradient-text">Made Simple</span>
  </h1>
  <p>
    Upload any image and our CNN model will classify it into one of 10 categories:
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Feature cards
# ---------------------------------------------------------
f1, f2, f3 = st.columns(3)
features = [
    ("⚡", "Instant Results", "Classify images with your trained CNN"),
    ("🛡️", "10 Categories", "Trained for the CIFAR-10 dataset"),
    ("🧠", "CNN Powered", "Deep convolutional neural network"),
]
for col, (icon, title, desc) in zip((f1, f2, f3), features):
    with col:
        st.markdown(
            f"""
            <div class="feature">
              <div class="feature-icon">{icon}</div>
              <div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.info("Keep cifar10_model.h5 in the same folder as app.py.")
    st.stop()

# ---------------------------------------------------------
# Main two-column area
# ---------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">📤 Upload Image</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="upload-card">
          <div class="upload-icon">🖼️</div>
          <div class="upload-title">Drop your image here</div>
          <div class="upload-text">JPG, JPEG or PNG • The model will resize it to 32×32</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except (UnidentifiedImageError, OSError):
            st.error("Invalid image file.")
            st.stop()

        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(f"Original size: {image.width} × {image.height} pixels")

        if st.button("🔄 Classify Again", use_container_width=True):
            st.rerun()

        # Store prediction in session state so it persists during reruns.
        file_key = f"{uploaded_file.name}-{uploaded_file.size}"
        if st.session_state.get("prediction_file") != file_key:
            with st.spinner("🧠 Running image through CNN layers..."):
                predictions = model.predict(preprocess(image), verbose=0)[0]
            st.session_state.prediction_file = file_key
            st.session_state.predictions = predictions

with right:
    st.markdown('<div class="section-title">🎯 Classification Result</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown(
            """
            <div class="confidence-box" style="min-height:360px;display:flex;
                 align-items:center;justify-content:center;text-align:center;">
              <div>
                <div style="font-size:45px;">🔮</div>
                <div style="color:rgba(255,255,255,.65);font-weight:700;margin-top:10px;">
                  Your prediction will appear here
                </div>
                <div style="color:rgba(255,255,255,.32);font-size:11px;margin-top:6px;">
                  Upload an image to start classification
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif "predictions" in st.session_state:
        predictions = np.asarray(st.session_state.predictions)
        top_idx = int(np.argmax(predictions))
        top_class = CLASS_NAMES[top_idx]
        top_confidence = float(predictions[top_idx]) * 100

        c1, c2 = CLASS_GRADIENTS[top_class]
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-inner"
                   style="background:linear-gradient(110deg,{c1},{c2});">
                <div class="prediction-row">
                  <div>
                    <div class="result-label">Top Prediction</div>
                    <div class="prediction-main">
                      <div class="prediction-emoji">{CLASS_EMOJIS[top_class]}</div>
                      <div>
                        <div class="prediction-name">{top_class}</div>
                        <div class="prediction-confidence">{top_confidence:.1f}% confidence</div>
                      </div>
                    </div>
                  </div>
                  <div class="confidence-circle">{top_confidence:.0f}%</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="confidence-box"><div class="confidence-heading">All Class Confidence</div>',
            unsafe_allow_html=True,
        )

        sorted_predictions = sorted(
            zip(CLASS_NAMES, predictions),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        for rank, (name, score) in enumerate(sorted_predictions):
            pct = float(score) * 100
            bar_css = gradient_css(name)
            top_class_css = "top" if rank == 0 else ""
            st.markdown(
                f"""
                <div class="class-row">
                  <div class="class-line">
                    <span class="class-name">{CLASS_EMOJIS[name]} &nbsp; {name}</span>
                    <span class="class-score {top_class_css}">{pct:.1f}%</span>
                  </div>
                  <div class="bar-bg">
                    <div class="bar" style="width:{min(pct,100):.2f}%;background:{bar_css};"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model architecture
# ---------------------------------------------------------
st.markdown(
    """
<div class="arch-card">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    <span style="font-size:20px;">🧠</span>
    <span style="color:rgba(255,255,255,.90);font-weight:700;font-size:14px;">
      Model Architecture
    </span>
  </div>
  <div class="arch-text">
    Your trained Keras CNN receives a 32×32 RGB image, normalizes pixel values,
    passes the image through the learned convolutional network, and returns
    probabilities for all 10 CIFAR-10 classes.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Show actual model information instead of hard-coding unsupported claims.
try:
    parameter_count = model.count_params()
    layer_count = len(model.layers)
except Exception:
    parameter_count = 0
    layer_count = 0

s1, s2, s3, s4 = st.columns(4)
stats = [
    ("Layers", str(layer_count)),
    ("Parameters", f"{parameter_count:,}"),
    ("Input Size", "32×32"),
    ("Classes", "10"),
]
for col, (value_label, value) in zip((s1, s2, s3, s4), stats):
    with col:
        st.markdown(
            f"""
            <div class="stat">
              <div class="stat-value">{value}</div>
              <div class="stat-label">{value_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# How it works
# ---------------------------------------------------------
st.markdown(
    """
<div class="how-card">
  <div style="color:rgba(255,255,255,.78);font-weight:700;font-size:14px;margin-bottom:22px;">
    How It Works
  </div>
</div>
""",
    unsafe_allow_html=True,
)

steps = [
    ("1", "📤", "Upload", "Select an image from your device"),
    ("2", "📐", "Resize", "Image is resized to 32×32 pixels"),
    ("3", "🧠", "Analyze", "CNN processes the image"),
    ("4", "🎯", "Predict", "Get class confidence scores"),
]
step_cols = st.columns(4)
for col, (num, icon, title, desc) in zip(step_cols, steps):
    with col:
        st.markdown(
            f"""
            <div style="text-align:center;padding:0 8px;">
              <div class="step-icon">{icon}</div>
              <div class="step-number">{num}</div>
              <div class="step-title">{title}</div>
              <div class="step-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# Classes
# ---------------------------------------------------------
selected = None
if "predictions" in st.session_state:
    selected = CLASS_NAMES[int(np.argmax(st.session_state.predictions))]

st.markdown(
    '<div class="classes-title">10 CIFAR-10 Classes</div>',
    unsafe_allow_html=True,
)

class_cols = st.columns(10)
for col, name in zip(class_cols, CLASS_NAMES):
    with col:
        selected_css = "selected" if name == selected else ""
        st.markdown(
            f"""
            <div class="class-chip {selected_css}">
              <div class="class-chip-emoji">{CLASS_EMOJIS[name]}</div>
              <div class="class-chip-name">{name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# Bottom information / sharing section
# ---------------------------------------------------------
st.markdown(
    """
<div class="share-card">
  <div style="font-size:42px;">✨</div>
  <div style="color:white;font-size:20px;font-weight:800;margin-top:6px;">
    Use Anywhere, Any Device
  </div>
  <div style="max-width:650px;margin:8px auto 0;color:rgba(255,255,255,.45);
              font-size:12px;line-height:1.7;">
    This Streamlit app runs directly in your browser. Deploy it to Streamlit Cloud
    and share the public link with anyone.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="footer">
  🧠 CIFAR-10 Classifier &nbsp; • &nbsp;
  Built with Python + TensorFlow/Keras + Streamlit &nbsp; • &nbsp;
  CNN Image Classification
</div>
""",
    unsafe_allow_html=True,
)
