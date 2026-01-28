import tensorflow as tf

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"SUCCESS! Found GPU: {gpus[0]}")
    print("Your 3080 is ready to train.")
    
    # Optional: Enable memory growth (prevents TF from hogging 100% VRAM instantly)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("ERROR: TensorFlow cannot see your GPU.")
    print("   - Did you install tensorflow==2.10?")
    print("   - Are you on Windows Native?")