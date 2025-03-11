import os
import shutil
import random
import numpy as np
import pydicom
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# ----------------------------
# File Upload & Model Configuration
# ----------------------------
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure the model path is correct and absolute for Azure compatibility
MODEL_PATH = os.path.join(os.getcwd(), "models", "BEST_CNN2.keras")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ----------------------------
# DICOM Preprocessing Function
# ----------------------------
def preprocess_dicom_images(dicom_paths):
    """
    Reads each DICOM file, resizes to (128,128),
    normalizes by dividing by 255.0, and reshapes to (N,128,128,1).
    """
    processed_images = []
    for path in dicom_paths:
        try:
            ds = pydicom.dcmread(path)
            image = ds.pixel_array
            image = cv2.resize(image, (128, 128))
            image = image.astype(np.float32) / 255.0
            processed_images.append(image)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return np.array(processed_images).reshape(-1, 128, 128, 1)

# ----------------------------
# Flask Application & Routes
# ----------------------------
app = Flask(__name__)

@app.route('/')
def home():
    """
    Serves the front-end HTML for file uploads.
    """
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Pancrea Safe | AI-Powered Pancreatic Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
            .container { max-width: 800px; margin: auto; padding: 20px; }
            .upload-section { margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Smart Pancreatic Analysis</h1>
            <p>Upload your DICOM files for AI-powered pancreatic cancer detection.</p>
            <form action="/predict" method="POST" enctype="multipart/form-data">
                <label for="dicom_files">Upload DICOM Files:</label><br>
                <input type="file" name="files" multiple><br><br>
                <button type="submit">Analyze</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'prediction': 'Model not available'})

    files = request.files.getlist('files')
    if not files:
        return jsonify({'prediction': 'No files uploaded'})

    temp_folder = os.path.join(UPLOAD_FOLDER, "upload_" + str(random.randint(1000, 9999)))
    os.makedirs(temp_folder, exist_ok=True)

    dicom_paths = []
    for file in files:
        if file.filename == '':
            continue
        full_path = os.path.join(temp_folder, file.filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        file.save(full_path)
        dicom_paths.append(full_path)

    processed_images = preprocess_dicom_images(dicom_paths)
    if len(processed_images) == 0:
        shutil.rmtree(temp_folder)
        return jsonify({'prediction': 'No valid DICOM files found'})

    predictions = model.predict(processed_images)
    avg_prediction = np.mean(predictions)
    prediction_text = "Tumor Detected" if avg_prediction > 0.5 else "No Tumor Detected"

    shutil.rmtree(temp_folder)
    return jsonify({'prediction': prediction_text})

if __name__ == '__main__':
    # Use the PORT environment variable set by Azure App Service
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
