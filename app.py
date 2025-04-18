




import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from PIL import Image


# Load your trained model
model = load_model("skin_cancer_model.h5")

# Image size used during training
IMG_SIZE = (64, 64)

# Preprocess uploaded image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Map prediction to label
def is_cancerous(prediction):
    return "🧬 **Cancerous (Malignant)**" if prediction > 0.5 else "✅ **Non-Cancerous (Benign)**"

# --- Streamlit UI ---
st.set_page_config(page_title="Skin Cancer Detector", page_icon="🔬", layout="centered")
st.markdown("""
    <style>
    /* Smooth fade-in */
    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translateY(20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Apply to all markdown elements */
    .stMarkdown {
        animation: fadeInUp 1s ease-in-out;
    }

    /* Stylish upload box */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        border: 1px solid #ccc;
        padding: 10px;
        transition: all 0.3s ease-in-out;
    }

    .stFileUploader:hover {
        box-shadow: 0 0 12px rgba(0, 0, 0, 0.15);
        transform: scale(1.01);
    }

    /* Stylish button */
    .stButton > button {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 0.6em 1.4em;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0px 6px 20px rgba(106, 17, 203, 0.4);
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0px 8px 24px rgba(37, 117, 252, 0.4);
    }

    /* Image 3D hover */
    img {
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        transition: transform 0.5s ease;
    }

    img:hover {
        transform: scale(1.03) rotateX(1deg) rotateY(1deg);
    }

    /* Center headings */
    h1, h2, h3, p {
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Confidence highlight */
    .confidence-score {
        color: #6a1b9a;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #6a1b9a;'>🔬 Skin Cancer Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a skin lesion image to check if it's <b>benign</b> or <b>malignant</b>.</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload image
uploaded_file = st.file_uploader("📁 Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    st.markdown("---")

    if st.button("🔍 Predict"):
        with st.spinner("🧠 Analyzing image..."):
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)[0][0]

            st.markdown("<h3 style='text-align: center; color: #00897b;'>🧪 Result</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center;'>{is_cancerous(prediction)}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>🔎 Confidence Score: <b>{prediction * 100:.2f}%</b></p>", unsafe_allow_html=True)
