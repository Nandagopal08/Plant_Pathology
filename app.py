import streamlit as st
from plant_disease_model import predict_image
import tempfile
import os

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱")

st.title("🌱 Plant Health Detector")
st.subheader("Upload a plant leaf image and find out if it's healthy or diseased!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    # Display the image using the new parameter
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        prediction = predict_image(tmp_path)
        st.success(prediction)

    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
