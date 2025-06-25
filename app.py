from flask import Flask, render_template, jsonify, request
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from werkzeug.utils import secure_filename
import os
import uuid
import torch.nn as nn
import torchvision.models as models
from collections import Counter
import numpy as np

app = Flask(__name__)

# Define the PretrainedResNet class
class PretrainedResNet(nn.Module):
    """A ResNet model with pretrained weights for image classification"""
    def __init__(self, num_classes):
        super().__init__()
        # Load pretrained ResNet18 model with updated 'weights' parameter
        self.network = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze all layers
        for param in self.network.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers (fine-tuning)
        for param in self.network.layer4.parameters():
            param.requires_grad = True

        # Modify the last fully connected layer to match the number of classes
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)

# Function to load the base models
def load_base_models(model_dir, num_classes, device, num_folds=5):
    base_models = []
    for i in range(1, num_folds + 1):
        model_path = os.path.join(model_dir, f'model_fold_{i}.pth')
        print(f"Loading base model from {model_path}...")
        model = PretrainedResNet(num_classes=num_classes).to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            base_models.append(model)
            print(f"Successfully loaded model from {model_path}.")
        except FileNotFoundError:
            print(f"Error: Model file {model_path} not found.")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
    return base_models

# Device Management
def get_default_device():
    """Pick GPU (CUDA) if available, else MPS (Apple), else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_default_device()
print(f"Using device: {device}")

# Paths Setup
model_dir = "/Users/gracegomes/Desktop/Demo"  

# Load base models
num_classes = 38
base_models = load_base_models(model_dir, num_classes=num_classes, device=device, num_folds=5)
if not base_models:
    raise Exception("No base models loaded. Exiting.")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class mappings
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___healthy',
    'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___healthy',
    'Corn___Northern_Leaf_Blight',
    'Grape___Black_rot',
    'Grape___Esca_(Black_measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper_bell___Bacterial_spot',
    'Pepper_bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___healthy',
    'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]


def preprocess_image(image_path):
    """Preprocess the image for model input"""
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def predict_image(image_path):
    """Predict the class of the input image using ensemble of base models"""
    image = preprocess_image(image_path)
    image = image.unsqueeze(0).to(device)  

    base_preds = []
    with torch.no_grad():
        for idx, model in enumerate(base_models):
            outputs = model(image)
            _, predicted_idx = torch.max(outputs, 1)
            base_preds.append(predicted_idx.item())
            print(f"Model {idx+1} prediction: {predicted_idx.item()} - {class_names[predicted_idx.item()]}")

    # Aggregate predictions (majority voting)
    vote_counts = Counter(base_preds)
    final_pred_idx, vote = vote_counts.most_common(1)[0]
    confidence = (vote / len(base_models)) * 100  

    # Get class name
    predicted_class = class_names[final_pred_idx]
    plant_name, disease = predicted_class.split('___')
    is_healthy = (disease.lower() == 'healthy')

    return {
        'plant_name': plant_name,
        'is_healthy': is_healthy,
        'disease': disease if not is_healthy else None,
        'confidence': confidence
    }


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle image prediction"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"  
            temp_dir = 'temp'
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, unique_filename)
            file.save(file_path)
            print(f"File saved to {file_path}")

            try:
                # Call the predict function
                result = predict_image(file_path)
                print(f"Prediction result: {result}")
            except Exception as e:
                print(f"Prediction failed: {e}")
                return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500
            finally:
                os.remove(file_path)
                print(f"Temporary file {file_path} removed.")

            return jsonify(result)

        else:
            return jsonify({'error': 'Unsupported file type'}), 400
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
