import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR = "data/test"
IMG_SIZE = 256
BATCH_SIZE = 8

# Transforms (NO augmentation)
tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45], std=[0.2])
])

test_ds = datasets.ImageFolder(TEST_DIR, transform=tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Load model
checkpoint = torch.load("weights/unip_resnet18.pth", map_location=DEVICE)
class_names = checkpoint["classes"]
num_classes = len(class_names)

model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())      

# Test Accuracy
accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
