import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import your_model  # Replace with your actual model import

# Load model
model = your_model.load_model()  # Replace with your own model loading
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("Dog vs Cat Image Classification")
st.title("you should only upload either cat or dog")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Confidence threshold (tune this value as needed)
CONFIDENCE_THRESHOLD = 0.7

if uploaded_file is not None:
    try:
        # Open and show image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            confidence = confidence.item()
            prediction = prediction.item()

        if confidence < CONFIDENCE_THRESHOLD:
            st.error("⚠️ Unable to confidently classify the image as a dog or cat. Please try a clearer image.")
        else:
            label = "Dog" if prediction == 0 else "Cat"  # Adjust based on your labels
            st.success(f"Prediction: **{label}** (Confidence: {confidence:.2f})")

    except UnidentifiedImageError:
        st.error("❌ The uploaded file is not a valid image. Please upload a JPG or PNG file.")
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {e}")
