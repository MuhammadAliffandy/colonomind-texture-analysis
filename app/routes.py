"""
app/routes.py
-------------
Flask web application routes for the Colonomind endoscopic image classifier.

Inference pipeline (per uploaded image):
    1. Preprocess image.
    2. Extract handcrafted features (20-dim).
    3. Scale handcrafted features.
    4. Compute UMAP coordinates (2-dim).
    5. Feed [image, scaled_feats, umap_feats] into multimodal Keras model.
    6. Extract Keras probability and dense_5 deep features (128-dim).
    7. Construct agent input (deep features + keras proba + entropy).
    8. Pass to LightGBM agent for final Mayo Score prediction.
    9. Return prediction, confidence scores, and feature details.
"""

import io
import os
import pickle
import sys
import traceback
import joblib
import scipy.stats
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, jsonify, request
from PIL import Image

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

ROUTES_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = ROUTES_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.texture_extractor import extract_glcm_dwt, _smart_preprocess  # noqa: E402

# ---------------------------------------------------------------------------
# Default artefact paths (overridable via environment variables)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH  = PROJECT_ROOT / "Model-colono" / "models-TryFindingBestModel.h5"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "Model-colono" / "models-scaler_handcrafted_20.pkl"
DEFAULT_UMAP_PATH   = PROJECT_ROOT / "Model-colono" / "models-umap_model_mixed.pkl"
DEFAULT_AGENT_PATH  = PROJECT_ROOT / "Model-colono" / "models-trained_feedback_agent_lightgbm_multimodal.pkl"

MODEL_PATH  = Path(os.environ.get("MODEL_PATH",  str(DEFAULT_MODEL_PATH)))
SCALER_PATH = Path(os.environ.get("SCALER_PATH", str(DEFAULT_SCALER_PATH)))
UMAP_PATH   = Path(os.environ.get("UMAP_PATH",   str(DEFAULT_UMAP_PATH)))
AGENT_PATH  = Path(os.environ.get("AGENT_PATH",  str(DEFAULT_AGENT_PATH)))

# Mayo Score class labels
MAYO_LABELS = {
    0: "Mayo 0 - Remission",
    1: "Mayo 1 - Mild",
    2: "Mayo 2 - Moderate",
    3: "Mayo 3 - Severe",
}

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

bp = Blueprint("inference", __name__)

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_keras_model():
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)

@lru_cache(maxsize=1)
def get_feature_extractor():
    keras_model = get_keras_model()
    return tf.keras.Model(
        inputs=keras_model.input,
        outputs=keras_model.get_layer("dense_5").output,
        name="feature_extractor"
    )

@lru_cache(maxsize=1)
def get_scaler():
    return joblib.load(str(SCALER_PATH))

@lru_cache(maxsize=1)
def get_umap_model():
    return joblib.load(str(UMAP_PATH))

@lru_cache(maxsize=1)
def get_lightgbm_agent():
    if AGENT_PATH.suffix == '.pkl':
        with open(str(AGENT_PATH), "rb") as f:
            return pickle.load(f)
    else:
        return lgb.Booster(model_file=str(AGENT_PATH))

# ---------------------------------------------------------------------------
# Inference Route
# ---------------------------------------------------------------------------

@bp.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Load and validate image
        img_bytes = file.read()
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return jsonify({"error": "Uploaded file is not a valid image"}), 400
            
        rgb_image = np.array(pil_img)
        
        # 1. Preprocess
        processed_img = _smart_preprocess(rgb_image)
        input_img = np.expand_dims(processed_img.astype(np.float32) / 255.0, axis=0)

        # 2. Handcrafted Features & UMAP
        raw_feats = extract_glcm_dwt(processed_img)
        scaler = get_scaler()
        umap_model = get_umap_model()
        
        scaled_feats = scaler.transform(np.array(raw_feats).reshape(1, -1))
        umap_feats = umap_model.transform(scaled_feats)
        
        # 3. Keras Multimodal Inference
        keras_model = get_keras_model()
        extractor = get_feature_extractor()
        
        keras_inputs = [input_img, scaled_feats, umap_feats]
        keras_proba = keras_model.predict(keras_inputs, verbose=0)[0]
        deep_features = extractor.predict(keras_inputs, verbose=0)[0]
        
        entropy = scipy.stats.entropy(keras_proba)
        
        # 4. Agent Input
        agent_input = np.hstack([
            deep_features.reshape(1, -1),
            keras_proba.reshape(1, -1),
            [[entropy]]
        ])
        
        # 5. Agent Prediction
        agent = get_lightgbm_agent()
        agent_proba = agent.predict(agent_input)
        
        # Handle LightGBM prediction formats
        if isinstance(agent_proba, (list, np.ndarray)):
            if len(agent_proba.shape) > 1 or len(agent_proba) > 1:
                # Multiclass output
                if len(agent_proba.shape) > 1:
                    agent_proba = agent_proba[0]
                agent_conf = float(np.max(agent_proba))
                agent_label_idx = int(np.argmax(agent_proba))
            else:
                agent_label_idx = int(agent_proba[0])
                agent_conf = 1.0
        else:
            agent_label_idx = int(agent_proba)
            agent_conf = 1.0

        # Create structured JSON response
        predicted_class_name = MAYO_LABELS.get(agent_label_idx, f"Unknown ({agent_label_idx})")
        
        CONFIDENCE_THRESHOLD = 0.70
        is_referral = agent_conf < CONFIDENCE_THRESHOLD
        status_msg = "Uncertainty Detected - Refer to Doctor" if is_referral else "High Confidence Analysis"
        
        feat_names = ["LL_Mean", "LL_Std", "LL_Var", "LL_Ent", "LH_Mean", "LH_Std", "LH_Var", "LH_Ent",
                      "HL_Mean", "HL_Std", "HL_Var", "HL_Ent", "HH_Mean", "HH_Std", "HH_Var", "HH_Ent", "HH_Energy",
                      "GLCM_Con", "GLCM_Dis", "GLCM_Hom"]
        feat_dict = {feat_names[i]: float(raw_feats[i]) for i in range(20)}
        
        response_data = {
            "prediction": agent_label_idx,
            "prediction_label": predicted_class_name,
            "confidence": float(agent_conf),
            "status": status_msg,
            "is_referral": is_referral,
            "features": {
                "texture": feat_dict,
                "umap": {
                    "x": float(umap_feats[0][0]),
                    "y": float(umap_feats[0][1])
                },
                "keras_proba": [float(p) for p in keras_proba],
                "entropy": float(entropy)
            }
        }

        return jsonify(response_data), 200

    except Exception as e:
        current_app.logger.error(f"Prediction error: {e}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error during prediction."}), 500

@bp.route("/health", methods=["GET"])
def health_check():
    try:
        get_keras_model()
        get_feature_extractor()
        get_scaler()
        get_umap_model()
        get_lightgbm_agent()
        return jsonify({"status": "ok", "message": "All models loaded successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"Model loading failed: {e}"}), 500

def create_app(test_config=None):
    from flask import Flask, render_template
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret"),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    )
    if test_config is not None:
        app.config.from_mapping(test_config)
        
    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")
        
    app.register_blueprint(bp, url_prefix="/api")
    return app
