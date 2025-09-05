# 🏥 Advanced Diabetic Foot Analysis System

This project is a web-based tool designed for the comprehensive analysis of diabetic foot conditions. It leverages a powerful dual-AI approach, combining a custom Deep Learning model for classification with advanced Large Language Models (from OpenAI) for detailed qualitative analysis and care recommendations.

The application provides a holistic assessment by analyzing images of both the left and right foot, delivering insights that are both quantitative and descriptive.

## 🚀 How It Works

The system follows a multi-step analysis pipeline to provide a comprehensive report:

1.  **Dual Image Upload**: The user uploads clear images of both the left and right foot through the interactive Streamlit interface.

2.  **Step 1: Deep Learning Classification (MobileNetV3)**:

      * A custom **PyTorch model** based on a Siamese-like architecture with **MobileNetV3** backbones processes both foot images simultaneously.
      * This model performs a binary classification to determine if the feet exhibit characteristics commonly associated with diabetes, outputting a prediction (**Diabetic** or **Non-Diabetic**) and a confidence score.

3.  **Step 2: Detailed Vision Analysis (GPT-4o-mini)**:

      * Each foot image is individually sent to the **OpenAI GPT-4o-mini Vision API**.
      * A highly structured prompt guides the AI to act as an expert podiatrist, performing a detailed examination of:
          * Skin Health (color, texture, calluses)
          * Circulation Indicators (swelling, discoloration)
          * Structural Deformities
          * Wounds or Ulcers
          * Nail Condition

4.  **Step 3: Actionable Recommendations (GPT-4)**:

      * The results from the classification model and the detailed vision analyses for both feet are aggregated.
      * This consolidated information is then sent to **OpenAI's GPT-4 model**.
      * Prompted as a senior diabetic foot specialist, GPT-4 generates practical and categorized care recommendations, including immediate actions, daily care protocols, and risk prevention strategies.

5.  **Step 4: Report Generation**:

      * All generated information—classification, detailed analyses, and recommendations—is compiled into a structured **JSON report** that users can download for their records.

## ✨ Key Features

  - **Dual-AI Pipeline**: Combines a custom CNN (MobileNetV3) for classification and LLMs (GPT-4o-mini, GPT-4) for nuanced analysis.
  - **Comparative Analysis**: The Siamese-like model architecture is designed to compare features between both feet.
  - **In-Depth Qualitative Insights**: Leverages GPT-4o-mini's vision capabilities for detailed, expert-level image description.
  - **Actionable Care Plans**: Generates structured and practical recommendations suitable for patient guidance.
  - **Interactive Web Interface**: Built with Streamlit for an easy-to-use and responsive user experience.
  - **Downloadable Reports**: Provides a comprehensive analysis report in JSON format.

-----

## 🛠️ Tech Stack

  - **Frontend**: [Streamlit](https://streamlit.io/)
  - **Deep Learning**: [PyTorch](https://pytorch.org/), [Torchvision](https://pytorch.org/vision/stable/index.html)
  - **LLM Integration**: [OpenAI API](https://www.google.com/search?q=https://openai.com/docs)
  - **Data Handling**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
  - **Image Processing**: [Pillow (PIL)](https://pillow.readthedocs.io/)

-----

## 🖥️ Setup and Installation

Follow these steps to run the application locally.

### 1\. Prerequisites

  - Python 3.8+
  - An OpenAI API Key

### 2\. Clone the Repository

```bash
git clone https://your-repository-url.git
cd your-repository-directory
```

### 3\. Install Dependencies

Create a `requirements.txt` file with the following content:

```
streamlit
pandas
torch
torchvision
openai
numpy
Pillow
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

### 4\. Add Model File

You must place the pre-trained PyTorch model file in the root directory of the project.

  - **Required file:** `mobilenet_v3_model.pth`

### 5\. Configure API Key

Set up your OpenAI API key using Streamlit's secrets management. Create a file at `.streamlit/secrets.toml` and add your key:

```toml
# .streamlit/secrets.toml

OPENAI_API_KEY = "your-openai-api-key-here"
```

### 6\. Run the Application

Execute the following command in your terminal:

```bash
streamlit run app.py
```

The application will open in your default web browser.

-----

## ⚠️ Disclaimer

This application is developed for medical research and informational purposes only. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always seek the advice of your physician or another qualified health provider with any questions you may have regarding a medical condition.
