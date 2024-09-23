import streamlit as st
import torch
import cv2
import numpy as np
from your_model_file import MobileNetV3Model  # Ganti dengan path file model Anda

# Fungsi preprocessing
def preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (224, 224))  # Resize to 224x224
    image = image / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
    image = torch.tensor(image, dtype=torch.float32)
    return image

# Fungsi untuk memuat model
def load_model():
    model = MobileNetV3Model(extractor_trainable=True)
    model.load_state_dict(torch.load('mobilenet_v3_model.pth'))
    model.eval()  # Set model ke mode evaluasi
    return model

# Muat model saat aplikasi dijalankan
model = load_model()

# Streamlit interface
st.title("Prediksi Penyakit Diabetes")

# Input data (upload gambar)
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca gambar dari file yang di-upload
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # Decode image

    # Preprocess gambar
    processed_image = preprocessing(image)

    # Lakukan prediksi
    with torch.no_grad():
        predictions = model(processed_image.unsqueeze(0))  # Tambahkan dimensi batch

    # Tampilkan hasil prediksi
    st.write("Hasil Prediksi:", predictions)

    # Menampilkan gambar yang di-upload
    st.image(image, channels="RGB", use_column_width=True, caption='Uploaded Image')
