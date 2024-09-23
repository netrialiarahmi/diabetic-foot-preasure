import streamlit as st
import pandas as pd
import torch
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import models
import torch.nn.functional as F

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
uploaded_left_image = st.file_uploader("Upload Left Foot Image", type=["jpg", "png"])
uploaded_right_image = st.file_uploader("Upload Right Foot Image", type=["jpg", "png"])

if uploaded_left_image and uploaded_right_image:
    left_image = preprocess_image(np.array(Image.open(uploaded_left_image)))
    right_image = preprocess_image(np.array(Image.open(uploaded_right_image)))
    
    left_image = left_image.unsqueeze(0).to('cpu')  # Add batch dimension
    right_image = right_image.unsqueeze(0).to('cpu')
    
    with torch.no_grad():
        prediction = model(left_image, right_image)
        st.write(f"Prediction: {'Diabetic' if prediction > 0.5 else 'Non-Diabetic'}")

