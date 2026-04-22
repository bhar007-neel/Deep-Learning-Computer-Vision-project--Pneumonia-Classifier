import os
import sys
from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)

# ----------------------------------------
# Device setup
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------
# Image preprocessing
# ----------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------------------
# Load model
# ----------------------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# model path from repo root
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "pneumonia_classifier.pth")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ----------------------------------------
# Prediction function
# ----------------------------------------
def predict_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return "PNEUMONIA" if pred.item() == 1 else "NORMAL"

# ----------------------------------------
# Routes
# ----------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        if "file" not in request.files:
            prediction = "No file uploaded"
        else:
            file = request.files["file"]

            if file.filename == "":
                prediction = "Please choose an image file"
            else:
                prediction = predict_image(file)

    return render_template("index.html", prediction=prediction)

# ----------------------------------------
# Run app
# ----------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)