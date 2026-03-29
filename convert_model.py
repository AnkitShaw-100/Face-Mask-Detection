import tensorflow as tf

print("[INFO] Loading model...")

model = tf.keras.models.load_model("mask_detector.model")

print("[INFO] Saving in .keras format...")

model.save("mask_detector.keras")

print("Done! Model converted successfully.")