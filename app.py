st.image(image, caption="Uploaded image", use_container_width=True)

with st.spinner("Predicting..."):
    resized = image.resize((32, 32))
    input_arr = preprocess(resized)
    predictions = model.predict(input_arr)[0]

top_idx = int(np.argmax(predictions))
top_class = CLASS_NAMES[top_idx]
top_conf = float(predictions[top_idx]) * 100

st.success(f"**Prediction: {top_class.upper()}** ({top_conf:.1f}% confidence)")

st.subheader("Confidence for all classes")
sorted_pairs = sorted(zip(CLASS_NAMES, predictions), key=lambda p: p[1], reverse=True)
for name, score in sorted_pairs:
    st.write(f"{name}")
    st.progress(float(score))
