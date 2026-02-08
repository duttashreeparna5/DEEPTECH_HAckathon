import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import cv2
import numpy as np
import os

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
WEIGHTS_PATH = "weights/unip_resnet18.pth"
OUTPUT_DIR = "gradcam_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD CHECKPOINT
# =========================
checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
class_names = checkpoint["classes"]
num_classes = len(class_names)

# =========================
# MODEL
# =========================
model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

# Target layer for Grad-CAM
target_layer = model.layer4[-1]

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45], std=[0.2])
])

# =========================
# GRAD-CAM FUNCTION
# =========================
def generate_gradcam(image_path):
    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    pred_idx = torch.argmax(output, dim=1).item()

    model.zero_grad()
    output[0, pred_idx].backward()

    h1.remove()
    h2.remove()

    act = activations[0].detach().cpu()
    grad = gradients[0].detach().cpu()

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam.numpy(), (IMG_SIZE, IMG_SIZE))

    # Convert original image to color
    orig = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    out_path = os.path.join(
        OUTPUT_DIR,
        f"gradcam_{class_names[pred_idx]}.png"
    )
    cv2.imwrite(out_path, overlay)

    print(f"Prediction: {class_names[pred_idx]}")
    print(f"Grad-CAM saved at: {out_path}")

