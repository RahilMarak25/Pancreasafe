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

# Make sure the 'models' folder exists and contains 'BEST_CNN2.keras'
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'BEST_CNN2.keras')
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
            /* (CSS omitted for brevity) */
        </style>
    </head>
    <body>
        <header class="header">
            <!-- Your header content -->
        </header>
        <div class="container">
            <div class="hero">
                <h1>Smart Pancreatic Analysis</h1>
                <p>Advanced AI detection with precision diagnostics</p>
            </div>
            <div class="upload-section">
                <label class="upload-label">Upload DICOM Folder for Analysis</label>
                <input type="file" id="fileInput" accept=".dcm" webkitdirectory directory multiple>
                <button class="custom-upload" onclick="document.getElementById('fileInput').click()">Select DICOM Folder</button>
                <button class="custom-upload" onclick="analyze()">Analyze DICOM Folder</button>
                <div id="preview" alt="Folder preview"></div>
                <div id="result"></div>
            </div>
            <!-- Additional sections as needed -->
        </div>
        <footer>
            <p>© 2024 Pancrea Safe. Advancing medical diagnostics through AI innovation.</p>
        </footer>
        <script>
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const preview = document.getElementById('preview');
                preview.style.display = 'block';
                let fileList = "<ul>";
                for (let i = 0; i < e.target.files.length; i++){
                    fileList += "<li>" + e.target.files[i].webkitRelativePath + "</li>";
                }
                fileList += "</ul>";
                preview.innerHTML = fileList;
            });

            function analyze() {
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                const preview = document.getElementById('preview');

                if (!fileInput.files.length) {
                    alert('Please select a DICOM folder first!');
                    return;
                }

                const formData = new FormData();
                for (let i = 0; i < fileInput.files.length; i++) {
                    formData.append('files', fileInput.files[i]);
                }

                resultDiv.style.display = 'flex';
                resultDiv.innerHTML = '<div class="loader"></div><span>Analyzing DICOM folder...</span>';
                resultDiv.className = '';

                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    resultDiv.innerHTML = data.prediction.includes('No')
                        ? '🎉 <span style="margin-left: 12px;">No Tumor Detected</span>'
                        : '⚠️ <span style="margin-left: 12px;">Tumor Detected</span>';
                    resultDiv.className = data.prediction.includes('No') ? 'no-tumor' : 'tumor';
                    preview.style.display = 'none';
                })
                .catch(error => {
                    resultDiv.innerHTML = '⚠️ Error processing DICOM folder';
                    resultDiv.className = 'tumor';
                });
            }
        </script>
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
    # Clean up any previous uploads
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Railway sets PORT as an environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
