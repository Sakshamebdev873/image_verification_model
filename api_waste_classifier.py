import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

# --- Configuration ---
# IMPORTANT: Update this path to where you saved your model file
MODEL_PATH = "models/waste_classifier_multilabel.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = sorted([
    "aerosol_cans","aluminum_food_cans","aluminum_soda_cans","cardboard_boxes",
    "cardboard_packaging","clothing","coffee_grounds","disposable_plastic_cutlery",
    "eggshells","food_waste","glass_beverage_bottles","glass_cosmetic_containers",
    "glass_food_jars","magazines","newspaper","office_paper","paper_cups",
    "plastic_cup_lids","plastic_detergent_bottles","plastic_food_containers",
    "plastic_shopping_bags","plastic_soda_bottles","plastic_straws",
    "plastic_trash_bags","plastic_water_bottles","shoes","steel_food_cans",
    "styrofoam_cups","styrofoam_food_containers","tea_bags"
])

# --- Model Loading ---
def load_model():
    """Load the pre-trained model from disk."""
    # 1. Re-create the model architecture
    model = models.resnet50(weights=None) # Don't load pretrained weights, we will load our own
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))

    # 2. Load the saved state dictionary
    # Use map_location to ensure the model loads correctly whether on CPU or GPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # 3. Set the model to evaluation mode
    model.eval()
    
    model = model.to(DEVICE)
    print(f"✅ Model loaded from {MODEL_PATH} and moved to {DEVICE}")
    return model

# --- Image Transformations ---
# This must be the SAME as the validation transform from your training script
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# --- Flask App Initialization ---
app = Flask(__name__)
model = load_model()
data_transform = get_transform()

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to classify a waste image."""
    # 1. Check if a file was sent
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    try:
        # 2. Read and process the image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # 3. Transform the image and add a batch dimension
        transformed_image = data_transform(image).unsqueeze(0)
        transformed_image = transformed_image.to(DEVICE)

        # 4. Make a prediction
        with torch.no_grad():
            outputs = model(transformed_image)
            # Apply sigmoid to get probabilities for multi-label classification
            probabilities = torch.sigmoid(outputs)
        
        # 5. Format the response
        predictions = []
        # Use a threshold to decide which classes are present
        threshold = 0.5 
        
        for i, prob in enumerate(probabilities[0]):
            if prob > threshold:
                predictions.append({
                    "label": CLASSES[i],
                    "probability": f"{prob.item():.4f}" # Format to 4 decimal places
                })

        # Sort predictions by probability for better readability
        predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server... Access at http://127.0.0.1:5000")
    # Use host='0.0.0.0' to make it accessible on your local network
    app.run(host='0.0.0.0', port=5000)