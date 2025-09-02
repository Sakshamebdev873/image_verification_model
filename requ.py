import requests
import os
import mimetypes # <--- IMPORT THE LIBRARY

# --- Configuration ---
# The URL of your running Flask API
API_URL = "http://127.0.0.1:5000/predict"

# IMPORTANT: Path to the image you want to test.
# You can now use a .jpg, .png, .webp, or other standard image file here.
IMAGE_PATH = "image_1.webp" # <-- CHANGE THIS to your file

# --- Main Script ---
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image file not found at '{IMAGE_PATH}'")
    print("Please update the IMAGE_PATH variable in this script.")
else:
    try:
        # --- NEW: Automatically determine the file's content type ---
        mime_type, _ = mimetypes.guess_type(IMAGE_PATH)
        # If the type can't be guessed, provide a generic default
        if mime_type is None:
            mime_type = 'application/octet-stream' 
        # -----------------------------------------------------------

        # Open the image file in binary mode
        with open(IMAGE_PATH, 'rb') as image_file:
            # The 'files' dictionary key 'file' must match the key in app.py
            # The hardcoded 'image/jpeg' is now replaced with the guessed mime_type
            files = {'file': (os.path.basename(IMAGE_PATH), image_file, mime_type)}
            
            print(f"Sending request to {API_URL} with image: {IMAGE_PATH}")
            print(f"Guessed Content-Type: {mime_type}")
            
            # Send the POST request
            response = requests.post(API_URL, files=files)

            # Check the response
            if response.status_code == 200:
                print("\n✅ Prediction successful!")
                print("Response JSON:")
                # Pretty print the JSON response
                import json
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"\n❌ Error: Received status code {response.status_code}")
                print("Response Text:", response.text)

    except requests.exceptions.ConnectionError as e:
        print("\n❌ Connection Error: Could not connect to the server.")
        print("Please make sure the Flask app ('app.py') is running.")