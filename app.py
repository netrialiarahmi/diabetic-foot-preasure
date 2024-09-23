import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models import mobilenet_v3_large

# Definisikan model MobileNetV3
class MobileNetV3Model(nn.Module):
    def __init__(self, extractor_trainable=True):
        super(MobileNetV3Model, self).__init__()
        self.model = mobilenet_v3_large(pretrained=True)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 1)  # Sesuaikan output

        if not extractor_trainable:
            for param in self.model.parameters():
                param.requires_grad = False  # Membekukan lapisan jika tidak trainable

    def forward(self, x):
        return self.model(x)

# Fungsi preprocessing
def preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32)
    return image

# Fungsi untuk memuat model
def load_model():
    model = MobileNetV3Model(extractor_trainable=True)
    model.load_state_dict(torch.load('mobilenet_v3_model.pth'))
    model.eval()
    return model

# Muat model saat aplikasi dijalankan
model = load_model()

# Streamlit interface
st.title("Prediksi Penyakit Diabetes")

# Input data (upload gambar)
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Preprocess gambar
    processed_image = preprocessing(image)

    # Lakukan prediksi
    with torch.no_grad():
        predictions = model(processed_image.unsqueeze(0))

    # Tampilkan hasil prediksi
    st.write("Hasil Prediksi:", predictions.item())

    # Menampilkan gambar yang di-upload
    st.image(image, channels="RGB", use_column_width=True, caption='Uploaded Image')
