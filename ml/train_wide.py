import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split

# --- CONFIG ---
DATASET_PATH = "../training_data/cremad/AudioWAV" 
SAVE_DIR = "../models"
MODEL_NAME = "voice_vitals_cremad_resnet_wide" # New Name

SAMPLE_RATE = 16000
INPUT_LEN = 48000 # 3 sec
BATCH_SIZE = 32 # Smaller batch size because the model is BIGGER (fits in VRAM)
EPOCHS = 100



# --- GPU SETUP ---
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"RUNNING ON GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("WARNING: Running on CPU.")

# CREMA-D Mapping
EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgust"]
NUM_CLASSES = len(EMOTIONS)

# --- DATA LOADER (Optimized) ---
def load_data(data_dir):
    X, y = [], []
    code_map = {"NEU":0, "HAP":1, "SAD":2, "ANG":3, "FEA":4, "DIS":5, "DES":5}
    
    print(f"Scanning {data_dir}...")
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".wav"): continue
            try:
                parts = file.split("_")
                if parts[2] in code_map:
                    label = code_map[parts[2]]
                    
                    # Load & Process immediately
                    path = os.path.join(root, file)
                    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                    
                    # Pad/Trim
                    if len(audio) > INPUT_LEN: audio = audio[:INPUT_LEN]
                    else: audio = np.pad(audio, (0, INPUT_LEN - len(audio)), 'constant')
                    
                    X.append(audio.reshape(-1, 1))
                    y.append(label)
            except: continue
    return np.array(X), np.array(y)

# --- THE WIDE ARCHITECTURE ---
def build_wide_resnet():
    inputs = layers.Input(shape=(INPUT_LEN, 1))
    
    # Stem: Start with 64 filters (Standard was 32)
    x = layers.Conv1D(64, 7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    # Deep & Wide ResBlocks
    # We go up to 512 filters to capture complex emotional texture
    for filters in [64, 128, 256, 512]:
        shortcut = layers.Conv1D(filters, 1, strides=2, padding='same')(x)
        
        x = layers.Conv1D(filters, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Dropout(0.2)(x) # Light dropout inside blocks
        x = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
        
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x) # Heavier dropout at the head
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name="ResNet_Wide")

if __name__ == "__main__":
    # 1. Setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)
    
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # 2. Load
    X, y = load_data(DATASET_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Build
    model = build_wide_resnet()
    model.summary()
    
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # 4. Train with Scheduler
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, f"{MODEL_NAME}.h5"), save_best_only=True),
        # Crucial: Slow down learning if we get stuck
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), 
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)

    # 5. Auto-Export
    print("Exporting TFLite...")
    model = models.load_model(os.path.join(SAVE_DIR, f"{MODEL_NAME}.h5"))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    with open(os.path.join(SAVE_DIR, f"{MODEL_NAME}.tflite"), "wb") as f:
        f.write(converter.convert())