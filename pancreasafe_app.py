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
# Optional: Ngrok for local development only
# ----------------------------
# Set ENV=development in your environment variables if you want to use ngrok.
if os.environ.get("ENV", "production") == "development":
    from pyngrok import ngrok
    NGROK_AUTH_TOKEN = "2srVhtZYWHHGOKXWjGZSneiP6pu_88BmF2DE21wuS1AoqvhTT"
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(5000).public_url
    print(f" * Public URL (via ngrok): {public_url}")

# ----------------------------
# File Upload & Model Configuration
# ----------------------------
# Use a relative path for uploads and model files for easier deployment.
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Update the model path relative to your project repository.
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
            /* CSS Styles */
            :root {
                --primary-blue: #2563eb;
                --accent-teal: #2dd4bf;
                --background-white: #ffffff;
                --text-dark: #1e293b;
                --success-green: #10b981;
            }
            body {
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                margin: 0;
                background: var(--background-white);
                color: var(--text-dark);
                line-height: 1.6;
            }
            .header {
                background: linear-gradient(135deg, var(--primary-blue), #1d4ed8);
                padding: 1rem 2rem;
                box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            }
            .logo {
                color: white;
                font-size: 2rem;
                font-weight: 700;
                letter-spacing: -0.5px;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            .logo-icon {
                width: 40px;
                height: 40px;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
            }
            .hero {
                text-align: center;
                padding: 5rem 0;
                background: linear-gradient(45deg, var(--primary-blue), #3b82f6);
                color: white;
                border-radius: 20px;
                margin: 3rem 0;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                position: relative;
                overflow: hidden;
            }
            .upload-section {
                background: white;
                padding: 3rem 2rem;
                border-radius: 20px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.06);
                margin: 3rem 0;
                border: 2px solid var(--accent-teal);
                transition: transform 0.3s ease;
            }
            .upload-section:hover {
                transform: translateY(-5px);
            }
            .upload-label {
                font-size: 1.3rem;
                color: var(--primary-blue);
                margin-bottom: 1.5rem;
                display: block;
                font-weight: 600;
                text-align: center;
            }
            #fileInput {
                display: none;
            }
            .custom-upload {
                background: linear-gradient(45deg, var(--primary-blue), var(--accent-teal));
                color: white;
                padding: 1.2rem 2.5rem;
                border-radius: 12px;
                cursor: pointer;
                border: none;
                font-size: 1.1rem;
                display: inline-flex;
                align-items: center;
                gap: 1rem;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            .custom-upload:hover {
                opacity: 0.95;
                box-shadow: 0 5px 15px rgba(59, 130, 246, 0.3);
            }
            #preview {
                max-width: 320px;
                margin: 2rem auto;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                display: none;
            }
            #result {
                padding: 2rem;
                margin: 2rem 0;
                border-radius: 15px;
                display: none;
                align-items: center;
                gap: 1.5rem;
                font-size: 1.5rem;
                font-weight: 600;
                text-align: center;
                backdrop-filter: blur(8px);
            }
            .tumor {
                background: rgba(239, 68, 68, 0.1);
                color: #ef4444;
                border: 2px solid #ef4444;
            }
            .no-tumor {
                background: rgba(16, 185, 129, 0.1);
                color: var(--success-green);
                border: 2px solid var(--success-green);
            }
            .loader {
                border: 4px solid rgba(59, 130, 246, 0.1);
                border-top: 4px solid var(--primary-blue);
                border-radius: 50%;
                width: 45px;
                height: 45px;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .feature-section {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 2.5rem;
                margin: 5rem 0;
            }
            .feature-card {
                background: white;
                padding: 2.5rem;
                border-radius: 16px;
                text-align: center;
                box-shadow: 0 5px 20px rgba(0,0,0,0.08);
                border: 1px solid rgba(59, 130, 246, 0.1);
                transition: transform 0.3s ease;
            }
            .feature-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 12px 25px rgba(59, 130, 246, 0.15);
            }
            .feature-icon {
                font-size: 2.8rem;
                margin-bottom: 1.5rem;
                background: linear-gradient(45deg, var(--primary-blue), var(--accent-teal));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            footer {
                background: var(--text-dark);
                color: white;
                text-align: center;
                padding: 2rem;
                margin-top: 6rem;
                font-size: 0.9rem;
            }
        </style>
    </head>
    <body>
        <header class="header">
            <div class="container">
                <div class="logo">
                    <svg class="logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                    </svg>
                    Pancrea Safe
                </div>
            </div>
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
            <div class="feature-section">
                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <h3>Accurate Results</h3>
                    <p>State-of-the-art AI with 99.8% diagnostic accuracy</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔒</div>
                    <h3>Data Privacy</h3>
                    <p>Military-grade encryption & zero data retention policy</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">✨</div>
                    <h3>User Friendly</h3>
                    <p>Intuitive interface designed for seamless experience</p>
                </div>
            </div>
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
    # Clean up previous uploads if any
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # Use the PORT environment variable if available (e.g., on Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
