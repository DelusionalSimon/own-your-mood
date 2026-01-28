import tensorflow as tf

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f" Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        try:
            details = tf.config.experimental.get_device_details(gpu)
            name = details.get('device_name', 'Unknown')
        except:
            name = "Unknown (Check Drivers)"
        
        print(f"   [ID: {i}] Name: {name} (Type: {gpu.device_type})")
    
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




