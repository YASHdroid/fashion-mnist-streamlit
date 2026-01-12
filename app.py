import streamlit as st
from fashion import predict_image

st.title("👕 Fashion MNIST Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    result = predict_image("temp.jpg")
    st.success(f"Predicted Class: {result}")

