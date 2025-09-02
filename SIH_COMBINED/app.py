import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

# --- 1. Configuration ---
MODEL_PATH = "models/waste_classifier_final_combined.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FIXED: The CLASSES list MUST have 32 items to match the trained model. ---
# IMPORTANT: Verify this list against the output from your training notebook.
# I have added 'other' and 'trash' as placeholders for the two missing classes.
CLASSES = sorted([
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes', 
    'cardboard_packaging', 'clothing', 'coffee_grounds', 'disposable_plastic_cutlery', 
    'eggshells', 'food_waste', 'glass_beverage_bottles', 'glass_cosmetic_containers', 
    'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups', 
    'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers', 
    'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws', 
    'plastic_trash_bags', 'plastic_water_bottles', 'shoes', 'steel_food_cans', 
    'styrofoam_cups', 'styrofoam_food_containers', 'tea_bags',
    'other', # Example of a missing class
    'trash'  # Example of a missing class
])
# -----------------------------------------------------------------------------

# --- 2. Environmental Data (add entries for your new classes if needed) ---
ENVIRONMENTAL_DATA = {
    # (Previous 30 entries are here) ...
    "aerosol_cans": {"disposal_category": "recycling", "info": "Ensure can is empty. Do not puncture. Recycle with metals."},
    "aluminum_food_cans": {"disposal_category": "recycling", "info": "Empty and rinse. Infinitely recyclable."},
    "aluminum_soda_cans": {"disposal_category": "recycling", "info": "Empty and rinse. Highly valuable for recycling."},
    "cardboard_boxes": {"disposal_category": "recycling", "info": "Flatten the box. Remove excessive plastic tape."},
    "cardboard_packaging": {"disposal_category": "recycling", "info": "Flatten. Discard greasy parts (e.g., from pizza)."},
    "glass_beverage_bottles": {"disposal_category": "recycling", "info": "Empty and rinse. Recycle caps separately."},
    "glass_cosmetic_containers": {"disposal_category": "recycling", "info": "Empty and rinse thoroughly. Remove pumps."},
    "glass_food_jars": {"disposal_category": "recycling", "info": "Empty and rinse. Recycle metal lids separately."},
    "magazines": {"disposal_category": "recycling", "info": "Recycle with other paper products."},
    "newspaper": {"disposal_category": "recycling", "info": "Recycle with mixed paper."},
    "office_paper": {"disposal_category": "recycling", "info": "Recycle in your paper bin. Staples are okay."},
    "plastic_detergent_bottles": {"disposal_category": "recycling", "info": "Usually #2 (HDPE) plastic. Empty, rinse, and put cap back on."},
    "plastic_soda_bottles": {"disposal_category": "recycling", "info": "#1 (PET) plastic. Empty, crush, and put cap back on."},
    "plastic_water_bottles": {"disposal_category": "recycling", "info": "#1 (PET) plastic. Empty, crush, and put cap back on."},
    "steel_food_cans": {"disposal_category": "recycling", "info": "Empty and rinse. Infinitely recyclable."},
    "coffee_grounds": {"disposal_category": "compost", "info": "Excellent for composting. Adds nitrogen to soil."},
    "eggshells": {"disposal_category": "compost", "info": "Great for compost piles. Adds calcium to soil."},
    "food_waste": {"disposal_category": "compost", "info": "Compost at home or use municipal organics collection to avoid landfill methane."},
    "tea_bags": {"disposal_category": "compost", "info": "Most are compostable, but check for plastic in the bag seal."},
    "clothing": {"disposal_category": "trash", "info": "Do not place in curbside recycling. Donate usable clothing first, otherwise trash."},
    "disposable_plastic_cutlery": {"disposal_category": "trash", "info": "Generally not recyclable due to size and plastic type."},
    "paper_cups": {"disposal_category": "trash", "info": "Most are not recyclable due to a plastic lining. Must be landfilled."},
    "plastic_cup_lids": {"disposal_category": "trash", "info": "Generally not recyclable. Check local rules for #5 or #6 plastic, but usually trash."},
    "plastic_food_containers": {"disposal_category": "trash", "info": "Recyclability varies. If not #1 or #2, it's often better to trash to avoid contamination."},
    "plastic_shopping_bags": {"disposal_category": "trash", "info": "DO NOT put in curbside bins; they jam machinery. Return to store drop-offs or trash."},
    "plastic_straws": {"disposal_category": "trash", "info": "Not recyclable. Too small for machinery."},
    "plastic_trash_bags": {"disposal_category": "trash", "info": "Not recyclable. Designed for landfill."},
    "shoes": {"disposal_category": "trash", "info": "Do not put in curbside recycling. Donate usable shoes first, otherwise trash."},
    "styrofoam_cups": {"disposal_category": "trash", "info": "Not recyclable in almost all programs."},
    "styrofoam_food_containers": {"disposal_category": "trash", "info": "Not recyclable and often contaminated with food."},
    # Add entries for the new classes
    "other": {"disposal_category": "trash", "info": "Item is unrecognized. Please dispose of in the trash."},
    "trash": {"disposal_category": "trash", "info": "This item should be placed in the trash."}
}

# --- 3. Model Loading and Transformations ---
def load_model():
    """Load the trained model from disk."""
    # This line now creates a model with 32 outputs, which will match the file
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    # This will now succeed because the shapes match
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    model.eval()
    model = model.to(DEVICE)
    print(f"✅ Model loaded from {MODEL_PATH} and moved to {DEVICE}")
    return model

def get_transform():
    """Get the image transformations."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# --- 4. Flask App Initialization ---
app = Flask(__name__)
model = load_model()
data_transform = get_transform()

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        transformed_image = data_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(transformed_image)
            probabilities = torch.sigmoid(outputs)
        
        raw_predictions = []
        threshold = 0.4
        for i, prob in enumerate(probabilities[0]):
            if prob > threshold:
                raw_predictions.append({"label": CLASSES[i], "probability": f"{prob.item():.4f}"})

        if not raw_predictions:
            return jsonify({"summary": "No waste items detected with high confidence.", "instructions": {}})

        response_data = {
            "summary": {"recycling": [], "compost": [], "trash": []},
            "instructions": {"recycling": set(), "compost": set(), "trash": set()}
        }
        
        for pred in raw_predictions:
            label = pred["label"]
            env_data = ENVIRONMENTAL_DATA.get(label)
            if env_data:
                category = env_data["disposal_category"]
                response_data["summary"][category].append(pred)
                response_data["instructions"][category].add(env_data["info"])

        final_instructions = {}
        for category, instruction_set in response_data["instructions"].items():
            if instruction_set:
                final_instructions[category] = " | ".join(sorted(list(instruction_set)))
            else:
                final_instructions[category] = f"No items detected for {category}."

        response_data["instructions"] = final_instructions

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server... Access at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)