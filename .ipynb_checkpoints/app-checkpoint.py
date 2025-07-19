import streamlit as st
from utils import preprocess_image, predict

# Page config with nice icon and layout
st.set_page_config(page_title="🐶🐱 Dog vs Cat Classifier", layout="wide", page_icon="🐾")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 2rem;
        border-radius: 15px;
    }
    .stFileUploader>div>div>input {
        border-radius: 10px;
    }
    .prediction {
        font-size: 24px;
        font-weight: 700;
        margin-top: 20px;
    }
    .confidence {
        font-size: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🐾 Dog vs Cat Image Classifier 🐾")

st.markdown(
    """
    Upload a clear image of either a **dog** or a **cat**, and our AI model will tell you which one it is with confidence.
    """
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Your Uploaded Image", use_column_width=True)

    with st.spinner("🐕🐈 Predicting..."):
        img_array = preprocess_image(uploaded_file)
        label, confidence = predict(img_array)
        
        # Map label to friendly names and emojis
        label_map = {
            0: ("🐱 Cat", "#FFA500"),  # orange color
            1: ("🐶 Dog", "#00BFFF"),  # sky blue color
        }
        pred_label, color = label_map.get(label, ("Unknown", "white"))

        st.markdown("---")
        st.markdown(
            f'<p class="prediction" style="color:{color};">Prediction: {pred_label}</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<p class="confidence" style="color:white;">Confidence: {confidence*100:.2f}%</p>',
            unsafe_allow_html=True
        )
st.markdown('</div>', unsafe_allow_html=True)
