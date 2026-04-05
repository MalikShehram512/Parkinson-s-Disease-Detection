import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Parkinson's AI Diagnostic",
    page_icon="🧠",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_my_model():
    # This is the correct, optimized way to load for Streamlit
    return tf.keras.models.load_model('parkinsons_model.h5')

model = load_my_model()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865322.png", width=100)
    st.title("Settings & Info")
    st.info("This AI analyzes Spiral and Wave drawings to detect tremors associated with Parkinson's Disease.")
    st.divider()
    st.write("🔍 **How to use:**")
    st.caption("1. Upload a clear top-down photo of the drawing.")
    st.caption("2. Ensure the drawing is on a plain white background.")
    st.caption("3. Click 'Run Diagnostic Analysis'.")

# --- Main UI ---
st.title("🧠 Parkinson's Disease Detection")
st.subheader("Early Screening AI Assistant")

col1, col2 = st.columns([1, 1])

with col1:
    st.write("### 📤 Upload Drawing")
    uploaded_file = st.file_uploader("Drop your image here (Spiral or Wave)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        # Professional touch: Add a border and center the image
        st.image(image, caption="Uploaded Specimen", use_container_width=True)

with col2:
    st.write("### 🔬 Diagnostic Analysis")
    
    if uploaded_file is None:
        st.warning("Please upload an image to begin analysis.")
    else:
        if st.button("Run Diagnostic Analysis"):
            with st.spinner("Analyzing patterns and micro-tremors..."):
                # 1. Preprocessing (Resize to match your training: 128x128)
                img = ImageOps.fit(image, (128, 128), Image.Resampling.LANCZOS)
                img_array = np.array(img.convert('RGB')) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Artificial delay for "Medical Analysis" feel
                time.sleep(1.5)
                
                # 2. Prediction
                prediction = model.predict(img_array)
                confidence = float(prediction[0][0])
                
                # 3. Interactive Result Display
                if confidence > 0.5:
                    st.markdown('<div class="prediction-box" style="background-color: #ffdce0; color: #af1921;">Parkinson\'s Indicators Detected</div>', unsafe_allow_html=True)
                    st.progress(confidence)
                    st.write(f"Confidence Level: **{confidence*100:.2f}%**")
                else:
                    st.markdown('<div class="prediction-box" style="background-color: #dcffe4; color: #1e7e34;">Healthy / No Indicators Detected</div>', unsafe_allow_html=True)
                    st.progress(1 - confidence)
                    st.write(f"Confidence Level: **{(1-confidence)*100:.2f}%**")

st.divider()
st.caption("⚠️ **Disclaimer:** This tool is for preliminary screening only. Please consult a medical professional for a formal diagnosis.")
