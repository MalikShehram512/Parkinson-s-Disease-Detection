import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Parkinson's AI Diagnostic",
    page_icon="🧠",
    layout="wide"
)

# --- 2. Custom Styling ---
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
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Optimized Model Loading (The FIX) ---
@st.cache_resource
def load_my_model():
    # This ensures the model is loaded only ONCE into memory
    return tf.keras.models.load_model('parkinsons_model.h5')

# Call the function once at the start
model = load_my_model()

# --- 4. Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2865/2865322.png", width=100)
    st.title("Settings & Info")
    st.info("This AI analyzes Spiral and Wave drawings to detect tremors associated with Parkinson's Disease.")
    st.divider()
    st.write("🔍 **How to use:**")
    st.caption("1. Upload a clear photo of the drawing.")
    st.caption("2. Ensure the background is plain white.")
    st.caption("3. Click 'Run Diagnostic Analysis'.")

# --- 5. Main UI ---
st.title("🧠 Parkinson's Disease Detection")
st.subheader("Early Screening AI Assistant")

col1, col2 = st.columns([1, 1])

with col1:
    st.write("### 📤 Upload Drawing")
    uploaded_file = st.file_uploader("Drop your image here (Spiral or Wave)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Specimen", use_container_width=True)

with col2:
    st.write("### 🔬 Diagnostic Analysis")
    
    if uploaded_file is None:
        st.warning("Please upload an image to begin analysis.")
    else:
        if st.button("Run Diagnostic Analysis"):
            with st.spinner("Analyzing patterns..."):
                # --- Preprocessing Steps ---
                # A. Resize to 128x128 (matches your CNN training)
                img = ImageOps.fit(image, (128, 128), Image.Resampling.LANCZOS)
                
                # B. Convert to RGB and Normalize
                img_array = np.array(img.convert('RGB')) / 255.0
                
                # C. Expand Dimensions (The FIX for ValueError)
                # Changes shape from (128, 128, 3) to (1, 128, 128, 3)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Small delay for UI feel
                time.sleep(1.0)
                
                # --- Prediction ---
                prediction = model.predict(img_array)
                confidence = float(prediction[0][0])
                
                # --- Result Display ---
                if confidence > 0.5:
                    st.markdown(
                        f'<div class="prediction-box" style="background-color: #ffdce0; color: #af1921;">'
                        f'Parkinson\'s Indicators Detected</div>', 
                        unsafe_allow_html=True
                    )
                    st.progress(confidence)
                    st.write(f"Confidence Level: **{confidence*100:.2f}%**")
                else:
                    st.markdown(
                        f'<div class="prediction-box" style="background-color: #dcffe4; color: #1e7e34;">'
                        f'Healthy / No Indicators Detected</div>', 
                        unsafe_allow_html=True
                    )
                    st.progress(1 - confidence)
                    st.write(f"Confidence Level: **{(1-confidence)*100:.2f}%**")

st.divider()
st.caption("⚠️ **Disclaimer:** This tool is for preliminary screening only. It is not a substitute for professional medical diagnosis.")
