import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# --- CONFIG ---
USE_MODEL = "RESNET"   # "TCN" or "RESNET"
DATASET   = "CREMAD"   # "RAVDESS" or "CREMAD"
SAMPLE_RATE = 16000
DURATION = 3 # seconds
INPUT_LEN = SAMPLE_RATE * DURATION
BATCH_SIZE = 64 
EPOCHS = 150    

# --- SAVE SETTINGS ---
SAVE_DIR = "../models"
BASE_FILENAME = "voice_model"
MODEL_PATH = "../models"

# Dataset Specifics
if DATASET == "RAVDESS":
    EMOTIONS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
    # Update this path if needed
    DEFAULT_DATA_PATH = "../training_data/archive" 
else: # CREMA-D
    EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgust"]
    # Update this path to where you extracted CREMA-D
    DEFAULT_DATA_PATH = "../training_data/cremad/AudioWAV" 

NUM_CLASSES = len(EMOTIONS)

# --- GPU SETUP ---
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

# --- DATA LOADING ---
def load_audio_file(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(audio) > INPUT_LEN:
            audio = audio[:INPUT_LEN]
        else:
            padding = INPUT_LEN - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        return audio.reshape(-1, 1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_dataset(data_dir):
    X, y = [], []
    print(f"Scanning {DATASET} folder: {data_dir}...")
    
    # Map for CREMA-D text codes to integers
    crema_map = {"NEU":0, "HAP":1, "SAD":2, "ANG":3, "FEA":4, "DIS":5, "DES":5}

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".wav"): continue
            
            try:
                label = None
                
                # --- LOGIC SWITCH ---
                if DATASET == "RAVDESS":
                    # Format: 03-01-06-01-02-01-12.wav (3rd part is emotion)
                    parts = file.split("-")
                    label = int(parts[2]) - 1 
                    
                elif DATASET == "CREMAD":
                    # Format: 1001_DFA_ANG_XX.wav (3rd part is code)
                    parts = file.split("_")
                    code = parts[2]
                    if code in crema_map:
                        label = crema_map[code]
                # --------------------

                if label is not None:
                    path = os.path.join(root, file)
                    audio = load_audio_file(path)
                    if audio is not None:
                        X.append(audio)
                        y.append(label)

            except (ValueError, IndexError):
                continue
                
    return np.array(X), np.array(y)

# --- MODEL 1: RESNET (Reliable) ---
def build_resnet_model():
    inputs = layers.Input(shape=(INPUT_LEN, 1))
    
    # Stem
    x = layers.Conv1D(32, 7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    # ResBlocks
    for filters in [64, 128, 256]:
        shortcut = layers.Conv1D(filters, 1, strides=2, padding='same')(x)
        x = layers.Conv1D(filters, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name="ResNet_Audio")

# --- MODEL 2: TCN (Experimental/High Tech) ---
def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.05):
    prev_x = x
    # Dilated Conv
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                      padding='same', dilation_rate=dilation_rate, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(dropout_rate)(x)
    
    # Second Conv
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                      padding='same', dilation_rate=dilation_rate, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(dropout_rate)(x)
    
    # Residual Match
    if prev_x.shape[-1] != filters:
        prev_x = layers.Conv1D(filters, 1, padding='same')(prev_x)
        
    x = layers.Add()([x, prev_x])
    return layers.ReLU()(x)

def build_tcn_model():
    inputs = layers.Input(shape=(INPUT_LEN, 1))
    
    # Stem (Aggressive Downsampling)
    x = layers.Conv1D(32, 11, strides=4, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    # TCN Stack
    x = tcn_block(x, 64, 3, 1)
    x = tcn_block(x, 64, 3, 2)
    x = tcn_block(x, 128, 3, 4)
    x = tcn_block(x, 128, 3, 8)
    x = tcn_block(x, 256, 3, 16)

    # Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name="TCN_Audio")

# --- MAIN ---
if __name__ == "__main__":
    # 1. LOAD DATA
    dataset_path = DEFAULT_DATA_PATH 
    
    if not os.path.exists(dataset_path):
        print(f"Error: Path not found: {dataset_path}")
        exit()

    X, y = load_dataset(dataset_path)
    print(f"Loaded {len(X)} files.")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. SELECT MODEL
    if USE_MODEL == "TCN":
        print("Building TCN Model...")
        model = build_tcn_model()
    else:
        print("Building ResNet Model...")
        model = build_resnet_model()
        
    model.summary()
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # 3. PREPARE SAVE PATHS
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Construct consistent filename: e.g. "voice_vitals_cremad_resnet"
    model_id = f"{BASE_FILENAME}_{DATASET.lower()}_{USE_MODEL.lower()}"
    
    # Define full paths
    h5_path = os.path.join(SAVE_DIR, f"{model_id}.h5")
    tflite_path = os.path.join(SAVE_DIR, f"{model_id}.tflite")

    # 4. CALLBACKS
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        # Save directly to the named h5 file instead of generic 'best_model.h5'
        callbacks.ModelCheckpoint(h5_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # 5. TRAIN
    print(f"Starting Training: {model_id}...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list
    )
    
    # 6. EXPORT
    print(f"Exporting to {tflite_path}...")
    # Reload the best weights from the specific h5 file
    best_model = models.load_model(h5_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"DONE! Files saved:\n - {h5_path}\n - {tflite_path}")