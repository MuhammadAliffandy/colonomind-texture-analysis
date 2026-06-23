import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

print("[INFO] Loading model...")
model = tf.keras.models.load_model("Model-colono/models-TryFindingBestModel.h5", compile=False)

print("\n--- MODEL INPUTS ---")
print(model.input)

print("\n--- MODEL SUMMARY ---")
model.summary()
