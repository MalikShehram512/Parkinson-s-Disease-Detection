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

# --- 3. Load the NEW Deployable Model ---
@st.cache_resource
def load_my_model():
    # Make sure this matches the exact name of the file you just uploaded!
    return tf.keras.models.load_model('parkinsons_model_deploy.h5')

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
                try:
                    # --- Preprocessing Steps for the New Model ---
                    # 1. Resize to exactly 128x128 
                    img = ImageOps.fit(image, (128, 128), Image.Resampling.LANCZOS)
                    
                    # 2. Convert to RGB (3 channels)
                    img = img.convert('RGB')
                    
                    # 3. Convert to Array and Normalize (0 to 1)
                    img_array = np.array(img) / 255.0
                    
                    # 4. Expand Dimensions from (128, 128, 3) to (1, 128, 128, 3)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Small delay for UI feel
                    time.sleep(1.0)
                    
                    # --- Prediction ---
                    prediction = model.predict(img_array)
                    
                    # Because we used class_mode='categorical', the output is a list of 2 probabilities:
                    # [Probability of Healthy (0), Probability of Parkinson's (1)]
                    parkinsons_confidence = float(prediction[0][1])
                    healthy_confidence = float(prediction[0][0])
                    
                    # --- Result Display ---
                    if parkinsons_confidence > 0.5:
                        st.markdown(
                            f'<div class="prediction-box" style="background-color: #ffdce0; color: #af1921;">'
                            f'Parkinson\'s Indicators Detected</div>', 
                            unsafe_allow_html=True
                        )
                        st.progress(parkinsons_confidence)
                        st.write(f"Confidence Level: **{parkinsons_confidence*100:.2f}%**")
                    else:
                        st.markdown(
                            f'<div class="prediction-box" style="background-color: #dcffe4; color: #1e7e34;">'
                            f'Healthy / No Indicators Detected</div>', 
                            unsafe_allow_html=True
                        )
                        st.progress(healthy_confidence)
                        st.write(f"Confidence Level: **{healthy_confidence*100:.2f}%**")
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.info("Please ensure your uploaded model is named 'parkinsons_model_deploy.h5'.")

st.divider()
st.caption("⚠️ **Disclaimer:** This tool is for preliminary screening only. It is not a substitute for professional medical diagnosis.")
