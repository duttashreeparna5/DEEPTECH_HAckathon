import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
WEIGHTS_PATH = "weights/unip_resnet18.pth"

# =========================
# LOAD CHECKPOINT
# =========================
checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
class_names = checkpoint["classes"]
num_classes = len(class_names)

# =========================
# MODEL (MATCH TRAINING)
# =========================
model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False
)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

# =========================
# TRANSFORMS (MATCH TRAIN/VAL)
# =========================
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45], std=[0.2])
])

# =========================
# PREDICT FUNCTION
# =========================
def predict(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    return class_names[pred_idx], probs[0][pred_idx].item()
