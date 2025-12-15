from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
import uuid

# Config Flask App
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"], methods=["GET","POST"]) # Izinkan semua origin (React frontend bisa request)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Pastikan folder uploads ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load AI Model
model_path = "chestXrayModelRevision.h5"
model = load_model(model_path)
print(f"Model loaded successfully! Path: {model_path}")

# Fungsi Prediksi
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image cannot be read")

    img = cv2.resize(img, (150, 150))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]  # probability Normal

    if pred >= 0.5:
        pred_class = "Normal"
        probability = pred  
    else:
        pred_class = "Pneumonia"
        probability = 1 - pred

    print(f"[DEBUG] {image_path} | raw_pred={pred:.4f} | class={pred_class} | prob={probability:.2f}%")
    print("Sending JSON:", {"prediction": pred_class, "probability": probability})

    return pred_class, float(probability)


# API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Simpan sementara file
    filename = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        pred_class, pred_prob = predict_image(file_path)
    except Exception as e:
        os.remove(file_path)
        return jsonify({'error': str(e)}), 500

    # Hapus file sementara
    os.remove(file_path)

    return jsonify({
        'prediction': pred_class,
        'probability': round(pred_prob, 2)
    })

# Test route
@app.route('/')
def home():
    return "Chest X-Ray AI Backend Running!"

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
