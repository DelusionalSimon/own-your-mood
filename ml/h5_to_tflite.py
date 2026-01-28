import os
import tensorflow as tf
from tensorflow.keras import models

# --- CONFIG ---
# Ensure this matches the name in your models folder
MODEL_NAME = "voice_model_cremad_resnet" 
MODELS_DIR = "../models"

H5_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.h5")
TFLITE_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.tflite")

def export():
    if not os.path.exists(H5_PATH):
        print(f"Error: Could not find {H5_PATH}")
        return

    print(f"Loading best checkpoint: {H5_PATH}...")
    model = models.load_model(H5_PATH)

    print("Converting to TFLite (Android format)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional: Enable default optimizations (makes it smaller)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print("SUCCESS!")
    print(f"File ready for Android: {TFLITE_PATH}")

if __name__ == "__main__":
    export()