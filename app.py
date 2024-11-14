import streamlit as st
import pandas as pd
import torch
import openai
import os
import numpy as np
import json
import base64
from PIL import Image
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from io import BytesIO

# Setup OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = openai_api_key

# MobileNetV3 Model Definition
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

def preprocess_image(image):
    """Preprocess image for model input"""
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image, dtype=torch.float32)

def analyze_image_with_openai(image, context=""):
    """Analyze image using OpenAI Vision API"""
    # Convert PIL Image to bytes
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    client = openai.OpenAI()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze this foot image in detail, considering:
                            1. Visual characteristics
                            2. Potential signs of diabetic conditions
                            3. Areas of concern
                            4. Recommendations for care
                            
                            Additional context: {context}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in image analysis: {str(e)}"

def main():
    st.title("Advanced Diabetic Foot Analysis System")
    st.write("Upload images of both feet for comprehensive analysis")

    # Load the model
    @st.cache_resource
    def load_model():
        model = MobileNetV3Model()
        model.load_state_dict(torch.load('mobilenet_v3_model.pth'))
        model.eval()
        return model

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Image upload
    uploaded_left_image = st.file_uploader("Upload Left Foot Image", type=["jpg", "png"])
    uploaded_right_image = st.file_uploader("Upload Right Foot Image", type=["jpg", "png"])

    if uploaded_left_image and uploaded_right_image:
        # Display images
        col1, col2 = st.columns(2)
        
        left_image = Image.open(uploaded_left_image)
        right_image = Image.open(uploaded_right_image)
        
        with col1:
            st.image(left_image, caption="Left Foot Image", width=300)
        
        with col2:
            st.image(right_image, caption="Right Foot Image", width=300)

        if st.button("Perform Comprehensive Analysis"):
            with st.spinner("Analyzing images..."):
                # 1. Model Prediction
                left_tensor = preprocess_image(left_image).unsqueeze(0)
                right_tensor = preprocess_image(right_image).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = model(left_tensor, right_tensor)
                    is_diabetic = prediction.item() > 0.5

                prediction_label = "Diabetic" if is_diabetic else "Non-Diabetic"
                prediction_probability = torch.sigmoid(prediction).item() * 100

                # Display prediction results
                st.write("### Classification Results")
                st.write(f"Prediction: **{prediction_label}**")
                st.write(f"Confidence: **{prediction_probability:.2f}%**")

                # 2. Detailed Visual Analysis
                st.write("### Detailed Analysis")
                
                # Analyze both images with OpenAI Vision
                left_analysis = analyze_image_with_openai(
                    left_image, 
                    f"This is the left foot image. Model prediction: {prediction_label}"
                )
                right_analysis = analyze_image_with_openai(
                    right_image,
                    f"This is the right foot image. Model prediction: {prediction_label}"
                )

                # Display analyses
                col3, col4 = st.columns(2)
                with col3:
                    st.write("#### Left Foot Analysis")
                    st.write(left_analysis)
                
                with col4:
                    st.write("#### Right Foot Analysis")
                    st.write(right_analysis)

                # 3. Generate Recommendations
                st.write("### Recommendations and Next Steps")
                recommendations_prompt = f"""
                Based on the analysis of both feet images and the classification as {prediction_label},
                please provide specific recommendations for:
                1. Immediate care steps
                2. Long-term management
                3. Lifestyle modifications
                4. When to seek medical attention
                """

                client = openai.OpenAI()
                recommendations_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a medical specialist in diabetic foot care."},
                        {"role": "user", "content": recommendations_prompt}
                    ]
                )
                
                st.write(recommendations_response.choices[0].message.content)

                # 4. Save Analysis
                if st.button("Download Analysis Report"):
                    report = {
                        "date": str(pd.Timestamp.now()),
                        "classification": {
                            "prediction": prediction_label,
                            "confidence": f"{prediction_probability:.2f}%"
                        },
                        "analysis": {
                            "left_foot": left_analysis,
                            "right_foot": right_analysis
                        },
                        "recommendations": recommendations_response.choices[0].message.content
                    }
                    
                    report_json = json.dumps(report, indent=4)
                    st.download_button(
                        label="Download JSON Report",
                        data=report_json,
                        file_name="foot_analysis_report.json",
                        mime="application/json"
                    )

if __name__ == "__main__":
    main()
