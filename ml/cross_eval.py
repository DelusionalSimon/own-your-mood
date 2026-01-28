import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import models
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIG ---
# Your "Gold Medal" Model
MODEL_PATH = "../models/voice_model_cremad_resnet.h5" 

# The "Foreign" Dataset (RAVDESS)
TEST_DATA_DIR = "../training_data/archive" 

SAMPLE_RATE = 16000
INPUT_LEN = 48000

# MAPPING: RAVDESS Index -> CREMA String
# RAVDESS: 0=Neu, 1=Calm, 2=Hap, 3=Sad, 4=Ang, 5=Fear, 6=Dis, 7=Surp
RAVDESS_LABELS = {
    0: "Neutral", 
    1: "Calm",      # Will be skipped
    2: "Happy", 
    3: "Sad", 
    4: "Angry", 
    5: "Fearful", 
    6: "Disgust", 
    7: "Surprised"  # Will be skipped
}

# CREMA Model Output Order
CREMA_CLASSES = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgust"]
CREMA_MAP = {name: i for i, name in enumerate(CREMA_CLASSES)}

def load_and_prep_ravdess(data_dir):
    X, y_true = [], []
    print(f"Scanning RAVDESS at {data_dir}...")
    
    cnt_skipped = 0
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".wav"): continue
            
            try:
                # Parse Filename: 03-01-06-01-02-01-12.wav
                parts = file.split("-")
                rav_idx = int(parts[2]) - 1 
                
                emotion_name = RAVDESS_LABELS.get(rav_idx)
                
                # FILTER: Only keep emotions the model knows
                if emotion_name not in CREMA_MAP:
                    cnt_skipped += 1
                    continue
                    
                target_class_id = CREMA_MAP[emotion_name]
                
                # Load Audio
                path = os.path.join(root, file)
                audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                
                # --- CRITICAL: NORMALIZE ---
                # RAVDESS is quiet. CREMA is loud. We must boost RAVDESS.
                #max_val = np.max(np.abs(audio))
                #if max_val > 0:
                 #   audio = audio / max_val
                # ---------------------------
                
                # Pad/Trim
                if len(audio) > INPUT_LEN:
                    audio = audio[:INPUT_LEN]
                else:
                    padding = INPUT_LEN - len(audio)
                    audio = np.pad(audio, (0, padding), 'constant')
                
                X.append(audio.reshape(INPUT_LEN, 1))
                y_true.append(target_class_id)

            except Exception as e:
                print(f"Error on {file}: {e}")
                continue
                
    print(f"Loaded {len(X)} files. (Skipped {cnt_skipped} incompatible files)")
    return np.array(X), np.array(y_true)

if __name__ == "__main__":
    # 1. Load Data
    X_test, y_test = load_and_prep_ravdess(TEST_DATA_DIR)
    
    # 2. Load Model
    print(f"Loading Model: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Check path.")
        exit()
        
    model = models.load_model(MODEL_PATH)
    
    # 3. Predict
    print("Running Inference on RAVDESS data...")
    y_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    
    # 4. Score
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*40)
    print(f"CROSS-CORPUS ROBUSTNESS SCORE: {acc*100:.2f}%")
    print("="*40)
    
    print("\nDetailed Breakdown:")
    print(classification_report(y_test, y_pred, target_names=CREMA_CLASSES))
    
    print("\nInterpreting the Score:")
    print(" - > 40%: EXCELLENT. Model understands human emotion, not just files.")
    print(" - 30-40%: GOOD. Acceptable for a hackathon demo.")
    print(" - < 25%: POOR. The model is overfitted to CREMA-D.")