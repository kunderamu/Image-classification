import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as transforms
import your_model  # Replace with your actual model import

# Load model
model = your_model.load_model()  # Replace with your own model loading
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("Binary Image Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Try to open the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()

        label = "Class A" if prediction == 0 else "Class B"  # Adjust based on your classes
        st.success(f"Prediction: **{label}**")

    except UnidentifiedImageError:
        st.error("❌ The uploaded file is not a valid image. Please upload a JPG or PNG file.")
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {e}")
