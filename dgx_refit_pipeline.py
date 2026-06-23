import os
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from app.texture_extractor import extract_handcrafted, _smart_preprocess

# Load models
print("[INFO] Loading models...")
keras_model = tf.keras.models.load_model("Model-colono/models-TryFindingBestModel.h5", compile=False)
scaler = joblib.load("Model-colono/models-scaler_handcrafted_20.pkl")
umap_model = joblib.load("Model-colono/models-umap_model_mixed.pkl")

# We want the output of dense_5 (128 dims)
concat_layer = "dense_5"
feature_extractor = tf.keras.Model(inputs=keras_model.input, outputs=keras_model.get_layer(concat_layer).output)

# Collect paths
limuc_root = Path("/raid/D13K48009/texture/LIMUC")
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
class_mapping = {
    "0": 0, "1": 1, "2": 2, "3": 3,
    "Mayo 0": 0, "Mayo 1": 1, "Mayo 2": 2, "Mayo 3": 3,
    "Mayo_0": 0, "Mayo_1": 1, "Mayo_2": 2, "Mayo_3": 3,
}

image_paths, labels = [], []
for ext in image_extensions:
    for img_path in limuc_root.rglob(f"*{ext}"):
        parent_name = img_path.parent.name
        if parent_name in class_mapping:
            image_paths.append(img_path)
            labels.append(class_mapping[parent_name])

n_images = len(image_paths)
print(f"[INFO] Found {n_images} images.")

all_dl_feats = []
all_texture_feats = []

batch_size = 32
for i in tqdm(range(0, n_images, batch_size), desc="Extracting Features"):
    batch_paths = image_paths[i:i+batch_size]
    
    batch_img = []
    batch_handcrafted = []
    
    for p in batch_paths:
        img_rgb = np.array(Image.open(p).convert("RGB"))
        processed_img = _smart_preprocess(img_rgb)
        
        # Keras image input
        input_img = (processed_img.astype(np.float32) / 255.0)
        batch_img.append(input_img)
        
        # Handcrafted features
        feats = extract_handcrafted(processed_img)
        batch_handcrafted.append(feats)
        
    batch_img = np.array(batch_img)
    batch_handcrafted = np.array(batch_handcrafted)
    
    # Scale and UMAP
    scaled_feats = scaler.transform(batch_handcrafted)
    umap_feats = umap_model.transform(scaled_feats)
    
    # Keras extract
    keras_inputs = [batch_img, scaled_feats, umap_feats]
    dl_feats = feature_extractor.predict(keras_inputs, verbose=0)
    
    all_dl_feats.append(dl_feats)
    all_texture_feats.append(batch_handcrafted)

if n_images > 0:
    all_dl_feats = np.vstack(all_dl_feats)
    all_texture_feats = np.vstack(all_texture_feats)
    
    np.save("limuc_dl_features.npy", all_dl_feats)
    np.save("limuc_texture_features.npy", all_texture_feats)
    np.save("limuc_labels.npy", np.array(labels))
    print("[INFO] Features saved for analysis!")
