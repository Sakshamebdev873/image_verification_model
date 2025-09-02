import requests
import os
import mimetypes
import json

# --- Configuration ---
API_URL = "http://127.0.0.1:5000/predict"

# IMPORTANT: Path to the image you want to test
IMAGE_PATH = "image_2.webp" # <-- CHANGE THIS to your image file

# --- Main Script ---
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image file not found at '{IMAGE_PATH}'")
else:
    try:
        mime_type, _ = mimetypes.guess_type(IMAGE_PATH)
        if mime_type is None:
            mime_type = 'application/octet-stream' 

        with open(IMAGE_PATH, 'rb') as image_file:
            files = {'file': (os.path.basename(IMAGE_PATH), image_file, mime_type)}
            
            print(f"Sending request to {API_URL} with image: {IMAGE_PATH}")
            
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                print("\n✅ Prediction successful!")
                print("Response JSON:")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"\n❌ Error: Received status code {response.status_code}")
                print("Response Text:", response.text)

    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Could not connect to the server.")
        print("Please make sure the Flask app ('app.py') is running.")