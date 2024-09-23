import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class MobileNetV3Model(nn.Module):
    def __init__(self, extractor_trainable=True):
        super(MobileNetV3Model, self).__init__()
        self.model = mobilenet_v3_large(pretrained=True)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 1)  # Sesuaikan output dengan jumlah kelas

        if not extractor_trainable:
            for param in self.model.parameters():
                param.requires_grad = False  # Membekukan lapisan jika tidak trainable

    def forward(self, x):
        return self.model(x)

# Jika ingin menjalankan file ini secara langsung (bukan saat di-import), Anda bisa menambahkan ini
if __name__ == "__main__":
    model = MobileNetV3Model()
    print(model)
