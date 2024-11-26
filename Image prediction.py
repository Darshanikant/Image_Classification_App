import streamlit as st
import numpy as np
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the pre-trained model
@st.cache_resource
def load_model():
    return EfficientNetV2B0(weights="imagenet")

model = load_model()


# Set background 
def set_bg(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg("https://cdn.photoroom.com/v2/image-cache?path=gs://background-7ef44.appspot.com/backgrounds_v3/black/16_-_black.jpg")

# App Details in Sidebar
st.sidebar.title("App Details")
st.sidebar.write("""
- **Name**: Image Classifier App  
- **Model**: EfficientNetV2B0  
- **Purpose**: Predict the object in an uploaded image.  
""")

# Main Header
st.title("Image Classification App")
st.write("Upload an image to predict its category!")

# File Upload Section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Load the image and preprocess it
    img = Image.open(uploaded_file)
    img_resized = img.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction Button
    if st.button("Predict"):
        st.write("Processing your image...")
        
        # Predict and Decode
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        # Display Predictions
        st.subheader("Top Predictions")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i + 1}: **{label}** ({score:.2f})")

        # Display Most Likely Prediction
        most_likely = decoded_predictions[0]
        st.success(f"It's a **{most_likely[1]}** image!")

# Footer
st.sidebar.write("Developed by Darshanikanta")
