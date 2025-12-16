# resave_model.py
from tensorflow.keras.models import load_model

# Path model lama (.h5) yang mungkin bikin error di TF 2.15
OLD_MODEL_PATH = "chestXrayModelRevision.h5"
NEW_MODEL_PATH = "chestXrayModelRevision_v2.h5"

# Load model lama tanpa compile
model = load_model(OLD_MODEL_PATH, compile=False)
print("Model loaded successfully from", OLD_MODEL_PATH)

# Save ulang agar compatible dengan TF 2.15
model.save(NEW_MODEL_PATH)
print("Model saved successfully to", NEW_MODEL_PATH)
