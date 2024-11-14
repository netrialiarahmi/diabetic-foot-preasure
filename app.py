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
import tempfile

# Page config
st.set_page_config(
    page_title="Diabetic Foot Analysis System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .main {
        padding: 2rem;
    }
    .diagnosis {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

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
    """Analyze image using OpenAI Vision API with improved prompting"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    client = openai.OpenAI()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert podiatrist and diabetic foot specialist with extensive experience in diabetic foot analysis. 
                    Your analysis should be:
                    1. Highly detailed and specific
                    2. Based on visible evidence in the image
                    3. Focused on diabetic-relevant indicators
                    4. Professional yet clear
                    5. Structured and methodical"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Please analyze this foot image in detail, focusing on:

1. Skin Health Assessment:
   - Color variations and patterns
   - Texture abnormalities
   - Dryness levels
   - Any breaks or damages
   - Presence of calluses

2. Circulation Indicators:
   - Color distribution patterns
   - Any signs of reduced blood flow
   - Presence of swelling
   - Temperature indicators (if visible)

3. Deformity Analysis:
   - Foot structure alignment
   - Pressure point locations
   - Joint positions and angles
   - Arch characteristics

4. Wound/Ulcer Inspection:
   - Presence of any wounds
   - Signs of healing or deterioration
   - Surrounding tissue condition
   - Infection indicators

5. Nail Condition:
   - Color and texture
   - Growth patterns
   - Signs of infection
   - Thickness abnormalities

Additional Context: {context}

Provide specific observations and measurements where possible. Avoid general statements and focus on actual visible features."""
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
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in image analysis: {str(e)}"

def generate_recommendations(classification_result, left_analysis, right_analysis):
    """Generate comprehensive recommendations based on analyses"""
    client = openai.OpenAI()
    
    recommendations_prompt = f"""Based on the following detailed foot analyses, provide specific care recommendations:

Classification: {classification_result}

Left Foot Analysis:
{left_analysis}

Right Foot Analysis:
{right_analysis}

Please provide detailed recommendations in these categories:

1. Immediate Actions Required:
   - Urgent care needs
   - Specific treatments
   - Professional consultations needed

2. Daily Care Protocol:
   - Cleaning procedures
   - Moisturizing recommendations
   - Inspection routine
   - Pressure relief methods

3. Risk Prevention Strategy:
   - Footwear recommendations
   - Activity modifications
   - Environmental considerations
   - Preventive measures

4. Monitoring Protocol:
   - What to check daily
   - Warning signs to watch
   - When to seek immediate care
   - Follow-up schedule

5. Lifestyle Adjustments:
   - Exercise recommendations
   - Dietary considerations
   - Daily activity modifications
   - Protective measures

Be specific, practical, and actionable in your recommendations."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior podiatrist specializing in diabetic foot care. Provide comprehensive, evidence-based recommendations that are practical and actionable."
                },
                {"role": "user", "content": recommendations_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

def main():
    st.title("🏥 Advanced Diabetic Foot Analysis System")
    st.markdown("""
    This system combines advanced AI models to analyze foot images and provide comprehensive diabetic foot assessments.
    Upload clear images of both feet for the most accurate analysis.
    """)

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

    # Create two columns for image upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Left Foot Image")
        uploaded_left_image = st.file_uploader("Upload Left Foot Image", type=["jpg", "png", "jpeg"])
        if uploaded_left_image:
            left_image = Image.open(uploaded_left_image)
            st.image(left_image, use_column_width=True)

    with col2:
        st.subheader("Right Foot Image")
        uploaded_right_image = st.file_uploader("Upload Right Foot Image", type=["jpg", "png", "jpeg"])
        if uploaded_right_image:
            right_image = Image.open(uploaded_right_image)
            st.image(right_image, use_column_width=True)

    if uploaded_left_image and uploaded_right_image:
        if st.button("Analyze Images", key="analyze_button"):
            with st.spinner("Analyzing images... Please wait."):
                try:
                    # 1. Model Prediction
                    left_tensor = preprocess_image(left_image).unsqueeze(0)
                    right_tensor = preprocess_image(right_image).unsqueeze(0)
                    
                    with torch.no_grad():
                        prediction = model(left_tensor, right_tensor)
                        is_diabetic = prediction.item() > 0.5

                    prediction_label = "Diabetic" if is_diabetic else "Non-Diabetic"
                    prediction_probability = torch.sigmoid(prediction).item() * 100

                    # Display prediction in a nice format
                    st.markdown("### 📊 Classification Results")
                    results_col1, results_col2 = st.columns(2)
                    with results_col1:
                        st.metric("Prediction", prediction_label)
                    with results_col2:
                        st.metric("Confidence", f"{prediction_probability:.1f}%")

                    # 2. Detailed Analysis
                    st.markdown("### 🔍 Detailed Analysis")
                    
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        st.markdown("#### Left Foot Analysis")
                        left_analysis = analyze_image_with_openai(
                            left_image, 
                            f"Left foot image. Model prediction: {prediction_label}"
                        )
                        st.write(left_analysis)
                    
                    with analysis_col2:
                        st.markdown("#### Right Foot Analysis")
                        right_analysis = analyze_image_with_openai(
                            right_image,
                            f"Right foot image. Model prediction: {prediction_label}"
                        )
                        st.write(right_analysis)

                    # 3. Recommendations
                    st.markdown("### 💡 Care Recommendations")
                    recommendations = generate_recommendations(
                        prediction_label,
                        left_analysis,
                        right_analysis
                    )
                    st.write(recommendations)

                    # 4. Generate Report
                    report = {
                        "date": str(pd.Timestamp.now()),
                        "classification": {
                            "prediction": prediction_label,
                            "confidence": f"{prediction_probability:.1f}%"
                        },
                        "analysis": {
                            "left_foot": left_analysis,
                            "right_foot": right_analysis
                        },
                        "recommendations": recommendations
                    }
                    
                    # Create JSON and PDF reports
                    report_json = json.dumps(report, indent=4)
                    
                    # Download buttons
                    st.markdown("### 📥 Download Reports")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download JSON Report",
                            data=report_json,
                            file_name="diabetic_foot_analysis_report.json",
                            mime="application/json"
                        )
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")

    # Add footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Developed for medical research purposes. Not a substitute for professional medical advice.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
