import streamlit as st
import pandas as pd
import torch
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
import openai

openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = openai_api_key

# Load your model and any other necessary data
class MobileNetV3Model(nn.Module):
    def __init__(self, extractor_trainable: bool = True):
        super(MobileNetV3Model, self).__init__()
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.feature_extractor = mobilenet.features
        self.fc = nn.Linear(mobilenet.classifier[0].in_features * 2, 1)

    def forward(self, left_image, right_image):
        x_left = self.feature_extractor(left_image)
        x_right = self.feature_extractor(right_image)
        x_left = F.adaptive_avg_pool2d(x_left, 1).reshape(x_left.size(0), -1)
        x_right = F.adaptive_avg_pool2d(x_right, 1).reshape(x_right.size(0), -1)
        x = torch.cat((x_left, x_right), dim=1)
        return self.fc(x)

model = MobileNetV3Model()
model.load_state_dict(torch.load('mobilenet_v3_model.pth'))
model.eval()

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image, dtype=torch.float32)

st.title("Diabetic Foot Classification")

# Upload images
uploaded_left_image = st.file_uploader("Upload Left Foot Image", type=["jpg", "png"])
uploaded_right_image = st.file_uploader("Upload Right Foot Image", type=["jpg", "png"])

if uploaded_left_image and uploaded_right_image:
    # Read and display images
    left_image = Image.open(uploaded_left_image)
    right_image = Image.open(uploaded_right_image)

    # Show image previews in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(left_image, caption="Left Foot Image", width=150)
    
    with col2:
        st.image(right_image, caption="Right Foot Image", width=150)

    st.markdown("---")

    # Add a Submit button
    if st.button("Analyze"):
        # Preprocess images for prediction
        left_image_tensor = preprocess_image(np.array(left_image))
        right_image_tensor = preprocess_image(np.array(right_image))

        left_image_tensor = left_image_tensor.unsqueeze(0).to('cpu')  # Add batch dimension
        right_image_tensor = right_image_tensor.unsqueeze(0).to('cpu')

        # Get prediction
        with torch.no_grad():
            prediction = model(left_image_tensor, right_image_tensor)
            is_diabetic = prediction > 0.5

        # Display prediction result
        prediction_label = "Diabetic" if is_diabetic else "Non-Diabetic"
        st.write(f"### Prediction: {prediction_label}")

        # Determine analysis prompt based on prediction
        analysis_prompt = (
            "You are an expert in diabetic foot analysis. "
            "The user has uploaded images of feet. "
            f"The prediction is that the patient is {prediction_label}. "
            "Please explain the implications of this diagnosis based on foot pressure maps, and describe the characteristics that support this conclusion."
        )
        client = openai.OpenAI(api_key=openai.api_key)
        # Call OpenAI API for analysis
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical assistant specializing in diabetic foot conditions."},
                {"role": "user", "content": analysis_prompt},
                {"role": "assistant", "content": ""}
            ]
        )

        analysis_result = response['choices'][0]['message']['content']

        st.write("### Analysis Result:")
        st.write(analysis_result)
