import os
import tensorflow as tf
from tensorflow.keras import models
import tf2onnx

# --- CONFIG ---
# Your best model (The CREMA one)
MODEL_PATH = "./models/voice_model_cremad_resnet.h5" 
ONNX_PATH  = "./models/voice_model_cremad_resnet.onnx"

def convert_to_onnx():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find {MODEL_PATH}")
        return

    print(f"Loading Keras model: {MODEL_PATH}...")
    model = models.load_model(MODEL_PATH)

    # Define input signature (Strictly required for ONNX)
    # [None, 48000, 1] -> Batch size is dynamic (None), length fixed
    spec = (tf.TensorSpec((None, 48000, 1), tf.float32, name="input_audio"),)

    print(f"Converting to ONNX (Opset 13)...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model, 
        input_signature=spec, 
        opset=13, # 13 is the "Goldilocks" version (compatible with almost everything)
        output_path=ONNX_PATH
    )
    
    print("✅ SUCCESS!")
    print(f"ONNX file ready for Embedl: {ONNX_PATH}")

if __name__ == "__main__":
    convert_to_onnx()