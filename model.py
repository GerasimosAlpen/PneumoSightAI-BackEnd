from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

MODEL_PATH = "chestXrayModelRevision.h5"
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading AI model...")
        _model = load_model(MODEL_PATH)
        print("Model loaded")
    return _model

def predict_image(image_path):
    model = get_model()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    if pred >= 0.5:
        return "Normal", float(pred)
    else:
        return "Pneumonia", float(1 - pred)
