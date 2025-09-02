import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

# --- Configuration ---
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

# --- NEW: Comprehensive Environmental Data Store for All 30 Classes ---
ENVIRONMENTAL_DATA = {
    "aerosol_cans": {
        "carbon_footprint_kg_co2e_per_item": 0.2,
        "recyclable": True,
        "recycling_info": "Ensure can is completely empty. Do not puncture or flatten. Place in recycling bin. The cap is often plastic and may need to be recycled separately.",
        "alternatives": "Use non-aerosol sprays or lotions. Buy products in pump bottles."
    },
    "aluminum_food_cans": {
        "carbon_footprint_kg_co2e_per_item": 0.015,
        "recyclable": True,
        "recycling_info": "Empty and rinse the can. The paper label can be left on. Aluminum is infinitely recyclable.",
        "alternatives": "Buy fresh or frozen foods. Cook in larger batches to reduce packaging."
    },
    "aluminum_soda_cans": {
        "carbon_footprint_kg_co2e_per_item": 0.013,
        "recyclable": True,
        "recycling_info": "Empty and rinse the can. Aluminum is one of the most valuable items in the recycling stream.",
        "alternatives": "Use a soda maker, drink from a fountain, or buy larger format bottles to reduce packaging per serving."
    },
    "cardboard_boxes": {
        "carbon_footprint_kg_co2e_per_kg": 0.6,
        "recyclable": True,
        "recycling_info": "Flatten the box. Remove excessive plastic tape and any non-paper packaging inserts.",
        "alternatives": "Reuse boxes for shipping or storage. Opt for products with minimal packaging."
    },
    "cardboard_packaging": {
        "carbon_footprint_kg_co2e_per_kg": 0.6,
        "recyclable": True,
        "recycling_info": "Flatten all packaging. If it's a food box (e.g., pizza box), discard any parts with grease or food residue as they cannot be recycled.",
        "alternatives": "Choose products with less packaging. Buy in bulk."
    },
    "clothing": {
        "carbon_footprint_kg_co2e_per_kg": 15.0, # Varies hugely by material
        "recyclable": False, # Not in curbside recycling
        "recycling_info": "Do not place in recycling bin. Donate usable clothing to thrift stores or charities. Find textile recycling drop-off points for damaged items.",
        "alternatives": "Buy second-hand, repair existing clothes, or choose durable, sustainably-made garments."
    },
    "coffee_grounds": {
        "carbon_footprint_kg_co2e_per_kg": 0.1, # Footprint is in the growing/transport
        "compostable": True,
        "recycling_info": "Excellent for composting. Adds nitrogen to soil. Many local coffee shops offer used grounds for free for gardens.",
        "alternatives": "N/A - This is a byproduct of consumption."
    },
    "disposable_plastic_cutlery": {
        "carbon_footprint_kg_co2e_per_item": 0.03,
        "recyclable": False,
        "recycling_info": "Generally not recyclable due to small size and type of plastic (often #6 Polystyrene). They fall through sorting machinery.",
        "alternatives": "Carry a reusable cutlery set (bamboo, steel). Opt for restaurants that use real silverware."
    },
    "eggshells": {
        "carbon_footprint_kg_co2e_per_kg": 0.05,
        "compostable": True,
        "recycling_info": "Great for compost piles. Crush them to decompose faster. They add calcium to the soil.",
        "alternatives": "N/A - This is a byproduct of consumption."
    },
    "food_waste": {
        "carbon_footprint_kg_co2e_per_kg": 2.5,
        "compostable": True,
        "recycling_info": "Do not put in recycling. Compost at home or use a municipal organics bin. In a landfill, food waste creates methane, a potent greenhouse gas.",
        "alternatives": "Plan meals to avoid over-buying. Store food properly. Eat leftovers."
    },
    "glass_beverage_bottles": {
        "carbon_footprint_kg_co2e_per_item": 0.3, # Heavier than plastic/aluminum
        "recyclable": True,
        "recycling_info": "Empty and rinse. Metal caps should be recycled separately. Labels can be left on.",
        "alternatives": "Use a reusable bottle. Buy drinks from a fountain."
    },
    "glass_cosmetic_containers": {
        "carbon_footprint_kg_co2e_per_item": 0.2,
        "recyclable": True,
        "recycling_info": "Empty and rinse thoroughly. Remove pumps or plastic components as they are not recyclable with the glass.",
        "alternatives": "Buy from brands that offer refills or use recycled/minimal packaging."
    },
    "glass_food_jars": {
        "carbon_footprint_kg_co2e_per_item": 0.25,
        "recyclable": True,
        "recycling_info": "Empty and rinse. Metal lids should be recycled separately with other metals.",
        "alternatives": "Reuse glass jars for food storage, organization, or canning."
    },
    "magazines": {
        "carbon_footprint_kg_co2e_per_kg": 1.2,
        "recyclable": True,
        "recycling_info": "Recycle with other paper products. No need to remove staples.",
        "alternatives": "Opt for digital subscriptions. Read content online."
    },
    "newspaper": {
        "carbon_footprint_kg_co2e_per_kg": 1.2,
        "recyclable": True,
        "recycling_info": "Recycle with mixed paper. Can also be used as compost liner, weed barrier, or for cleaning windows.",
        "alternatives": "Read news from digital sources."
    },
    "office_paper": {
        "carbon_footprint_kg_co2e_per_kg": 1.3,
        "recyclable": True,
        "recycling_info": "Recycle in your paper bin. Staples are okay. Avoid shredding if possible as it reduces fiber length.",
        "alternatives": "Go paperless. Print double-sided. Use digital documents and signatures."
    },
    "paper_cups": {
        "carbon_footprint_kg_co2e_per_item": 0.1,
        "recyclable": False, # In most places
        "recycling_info": "Most paper cups are not recyclable due to a thin plastic (polyethylene) lining. Check for specific local programs, but usually must be landfilled.",
        "alternatives": "Use a reusable coffee cup. Many cafes offer a discount for bringing your own."
    },
    "plastic_cup_lids": {
        "carbon_footprint_kg_co2e_per_item": 0.02,
        "recyclable": "Check Locally",
        "recycling_info": "Often made of #5 (PP) or #6 (PS) plastic. Check the number and your local recycling guidelines. Many places do not accept them.",
        "alternatives": "Go lidless if safe to do so. Use a reusable cup with its own lid."
    },
    "plastic_detergent_bottles": {
        "carbon_footprint_kg_co2e_per_item": 0.2,
        "recyclable": True,
        "recycling_info": "Usually made from #2 (HDPE) plastic, which is widely recycled. Empty, rinse, and put the cap back on.",
        "alternatives": "Use concentrated detergent strips or powder in cardboard boxes. Buy from refill stores."
    },
    "plastic_food_containers": {
        "carbon_footprint_kg_co2e_per_item": 0.15,
        "recyclable": "Check Locally",
        "recycling_info": "Check the number on the bottom (#1, #2, #5 are more commonly recycled). Must be empty, clean, and dry. Flimsy containers are less likely to be recycled.",
        "alternatives": "Use reusable glass or metal food containers. Buy in bulk to reduce packaging."
    },
    "plastic_shopping_bags": {
        "carbon_footprint_kg_co2e_per_item": 0.01,
        "recyclable": False, # Not in curbside recycling
        "recycling_info": "DO NOT put in your home recycling bin; they jam machinery. Return clean and dry bags to designated store drop-off bins.",
        "alternatives": "Use reusable shopping bags made from cloth or other durable materials."
    },
    "plastic_soda_bottles": {
        "carbon_footprint_kg_co2e_per_item": 0.08,
        "recyclable": True,
        "recycling_info": "Made from #1 (PET) plastic, which is highly recyclable. Empty the bottle, crush it to save space, and put the cap back on.",
        "alternatives": "Use a soda maker or buy drinks from a fountain."
    },
    "plastic_straws": {
        "carbon_footprint_kg_co2e_per_item": 0.001,
        "recyclable": False,
        "recycling_info": "Not recyclable. They are too small and lightweight to be sorted by recycling machinery.",
        "alternatives": "Decline a straw. Use a reusable straw made of metal, silicone, or bamboo."
    },
    "plastic_trash_bags": {
        "carbon_footprint_kg_co2e_per_item": 0.04,
        "recyclable": False,
        "recycling_info": "Not recyclable. Their purpose is to contain waste for the landfill.",
        "alternatives": "Reduce your overall waste to use fewer bags. Some bags are made from recycled content."
    },
    "plastic_water_bottles": {
        "carbon_footprint_kg_co2e_per_item": 0.08,
        "recyclable": True,
        "recycling_info": "Made from #1 (PET) plastic. Empty, crush, and replace the cap before recycling.",
        "alternatives": "Use a reusable water bottle and fill it from the tap."
    },
    "shoes": {
        "carbon_footprint_kg_co2e_per_pair": 14.0, # Average for running shoes
        "recyclable": False, # Not in curbside recycling
        "recycling_info": "Do not place in recycling bin. Donate usable shoes. Some brands (like Nike) have take-back programs to recycle them into new materials.",
        "alternatives": "Buy durable, well-made shoes. Repair them when possible."
    },
    "steel_food_cans": {
        "carbon_footprint_kg_co2e_per_item": 0.018,
        "recyclable": True,
        "recycling_info": "Empty and rinse. Steel is infinitely recyclable. You can place the lid inside the can for safety.",
        "alternatives": "Choose fresh, frozen, or glass-packaged alternatives."
    },
    "styrofoam_cups": {
        "carbon_footprint_kg_co2e_per_item": 0.05,
        "recyclable": False, # In most places
        "recycling_info": "Styrofoam (Polystyrene #6) is not recyclable in the vast majority of municipal programs. It is bulky, lightweight, and contaminates other materials.",
        "alternatives": "Use a reusable cup. If you must use disposable, choose a paper cup (though still not ideal)."
    },
    "styrofoam_food_containers": {
        "carbon_footprint_kg_co2e_per_item": 0.1,
        "recyclable": False, # In most places
        "recycling_info": "Not recyclable in most programs. Often contaminated with food, making it impossible to recycle.",
        "alternatives": "Bring your own container for leftovers. Support restaurants that use compostable or recyclable packaging."
    },
    "tea_bags": {
        "carbon_footprint_kg_co2e_per_item": 0.005,
        "compostable": True, # Usually
        "recycling_info": "Most tea bags are compostable, but some contain plastic (polypropylene) to seal them. Check the brand. Remove any staples before composting.",
        "alternatives": "Use loose-leaf tea with a reusable infuser."
    }
}


# --- Model Loading ---
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model = model.to(DEVICE)
    print(f"✅ Model loaded from {MODEL_PATH} and moved to {DEVICE}")
    return model

# --- Image Transformations ---
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
        
        # --- Combine model predictions with environmental data ---
        predictions = []
        threshold = 0.5 
        for i, prob in enumerate(probabilities[0]):
            if prob > threshold:
                label = CLASSES[i]
                # Look up the environmental data for the predicted label
                env_data = ENVIRONMENTAL_DATA.get(label, {}) # Use .get() for safety
                
                # Create the full response object for this prediction
                prediction_response = {
                    "label": label,
                    "probability": f"{prob.item():.4f}",
                    "environmental_info": env_data
                }
                predictions.append(prediction_response)
        
        # Sort predictions by probability for better readability
        predictions = sorted(predictions, key=lambda x: float(x['probability']), reverse=True)
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server... Access at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)