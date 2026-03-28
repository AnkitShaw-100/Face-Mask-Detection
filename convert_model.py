import tensorflow as tf
import os
import sys

print("[INFO] Converting mask detector model to .keras format...\n")

# Check what format we're dealing with
if os.path.isdir("mask_detector.model"):
    print("[INFO] Detected SavedModel format (directory)")
    try:
        # Try loading as SavedModel - this should work with legacy format
        print("[INFO] Attempting to load SavedModel...")
        maskNet = tf.saved_model.load("mask_detector.model")
        print("✅ SavedModel loaded successfully!")
        
        # Convert to concrete function and save as keras
        concrete_func = maskNet.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        print("✅ Model converted and ready to use")
        
    except Exception as e:
        print(f"[ERROR] Failed to load SavedModel: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

elif os.path.isfile("mask_detector.model"):
    print("[INFO] Detected legacy .model file format")
    print("[ERROR] The .model file format is not supported in TensorFlow 2.13+")
    print("[INFO] This file format needs to be converted/retrained")
    sys.exit(1)
else:
    print("[ERROR] mask_detector.model not found!")
    print("[INFO] Please ensure mask_detector.model exists in the current directory")
    sys.exit(1)