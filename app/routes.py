"""
app/routes.py
-------------
Flask web application routes for the Colonomind endoscopic image classifier.

Inference pipeline (per uploaded image):
    1. Preprocess image.
    2. Extract deep-learning (DL) feature vector from the frozen Keras backbone.
    3. Extract GLCM + DWT texture feature vector via texture_extractor.
    4. Concatenate DL + Texture features.
    5. Scale the concatenated vector with the fitted StandardScaler.
    6. Produce a 3-D UMAP embedding for the visualisation layer.
    7. Pass the scaled full-dimensional feature vector to the LightGBM agent
       (new_model/agent_deep_verified.txt) for Mayo Score prediction.
    8. Return prediction, confidence scores, texture metrics, and UMAP
       coordinates as a JSON response.

Environment variables (optional overrides):
    MODEL_PATH   – path to best_model.h5
    SCALER_PATH  – path to scaler_final.pkl
    UMAP_PATH    – path to umap_final.pkl
    AGENT_PATH   – path to agent_deep_verified.txt

Dependencies:
    flask>=2.3
    tensorflow>=2.12
    scikit-learn>=1.3
    umap-learn>=0.5
    lightgbm>=4.0
    Pillow>=9.0
    numpy>=1.23
    pywt (PyWavelets)>=1.4
    scikit-image>=0.21
"""

import io
import os
import pickle
import sys
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, jsonify, request
from PIL import Image

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

ROUTES_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = ROUTES_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.texture_extractor import (  # noqa: E402
    N_TEXTURE_FEATURES,
    extract_glcm_dwt,
)

# ---------------------------------------------------------------------------
# Default artefact paths (overridable via environment variables)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH  = PROJECT_ROOT / "new_model" / "best_model.h5"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "new_model" / "scaler_final.pkl"
DEFAULT_UMAP_PATH   = PROJECT_ROOT / "new_model" / "umap_final.pkl"
DEFAULT_AGENT_PATH  = PROJECT_ROOT / "new_model" / "agent_deep_verified.txt"

MODEL_PATH  = Path(os.environ.get("MODEL_PATH",  str(DEFAULT_MODEL_PATH)))
SCALER_PATH = Path(os.environ.get("SCALER_PATH", str(DEFAULT_SCALER_PATH)))
UMAP_PATH   = Path(os.environ.get("UMAP_PATH",   str(DEFAULT_UMAP_PATH)))
AGENT_PATH  = Path(os.environ.get("AGENT_PATH",  str(DEFAULT_AGENT_PATH)))

# Target image size expected by the Keras backbone
TARGET_IMAGE_SIZE = (224, 224)

# Mayo Score class labels
MAYO_LABELS = {
    0: "Mayo 0 — Remission",
    1: "Mayo 1 — Mild",
    2: "Mayo 2 — Moderate",
    3: "Mayo 3 — Severe",
}

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

bp = Blueprint("inference", __name__)

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# (loaded once on first request; avoids startup cost when running tests)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_feature_extractor():
    """
    Load the Keras model and return a feature-extractor sub-model.
    The classification head (final Dense/Softmax layer) is stripped.
    Weights are loaded from MODEL_PATH and frozen (inference-only).

    Returns
    -------
    tuple[keras.Model, int]
        (feature_extractor, n_dl_features)
    """
    import tensorflow as tf  # noqa: PLC0415

    current_app.logger.info(f"Loading Keras model from: {MODEL_PATH}")
    full_model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    full_model.trainable = False

    # Find the penultimate Dense layer (last layer before classification head)
    feature_layer = None
    for layer in reversed(full_model.layers):
        if (
            isinstance(layer, tf.keras.layers.Dense)
            and layer.name != full_model.layers[-1].name
        ):
            feature_layer = layer
            break

    if feature_layer is None:
        feature_layer = full_model.layers[-2]
        current_app.logger.warning(
            f"Falling back to layer '{feature_layer.name}' as feature layer."
        )
    else:
        current_app.logger.info(
            f"Feature extraction layer: '{feature_layer.name}'"
        )

    feature_extractor = tf.keras.Model(
        inputs=full_model.input,
        outputs=feature_layer.output,
        name="feature_extractor",
    )
    n_dl_features = int(feature_extractor.output_shape[-1])
    current_app.logger.info(f"DL feature dimensionality: {n_dl_features}")
    return feature_extractor, n_dl_features


@lru_cache(maxsize=1)
def _get_scaler():
    """Load and return the fitted StandardScaler from SCALER_PATH."""
    current_app.logger.info(f"Loading scaler from: {SCALER_PATH}")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return scaler


@lru_cache(maxsize=1)
def _get_umap():
    """Load and return the fitted UMAP reducer from UMAP_PATH."""
    current_app.logger.info(f"Loading UMAP reducer from: {UMAP_PATH}")
    with open(UMAP_PATH, "rb") as f:
        umap_reducer = pickle.load(f)
    return umap_reducer


@lru_cache(maxsize=1)
def _get_lgb_agent():
    """
    Load and return the LightGBM booster from the text-format agent file.

    The agent file (agent_deep_verified.txt) contains the serialised LightGBM
    model in the native LightGBM text format. It is loaded via
    lightgbm.Booster(model_file=...).

    Returns
    -------
    lightgbm.Booster
    """
    import lightgbm as lgb  # noqa: PLC0415

    current_app.logger.info(f"Loading LightGBM agent from: {AGENT_PATH}")
    booster = lgb.Booster(model_file=str(AGENT_PATH))
    return booster


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _read_image(file_storage) -> np.ndarray:
    """
    Read an uploaded file and return a uint8 RGB numpy array of shape (H, W, 3).

    Parameters
    ----------
    file_storage : werkzeug.datastructures.FileStorage
        Uploaded file object from Flask request.

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype uint8
    """
    img_bytes = file_storage.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _preprocess_for_dl(rgb_array: np.ndarray) -> np.ndarray:
    """
    Normalise a uint8 RGB image to a float32 batch tensor for the Keras model.

    Parameters
    ----------
    rgb_array : np.ndarray, shape (H, W, 3), dtype uint8

    Returns
    -------
    np.ndarray, shape (1, H, W, 3), dtype float32, values in [0, 1]
    """
    arr = rgb_array.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def _run_inference(rgb_image: np.ndarray) -> dict:
    """
    Execute the full hybrid inference pipeline for a single image.

    Pipeline:
        1. DL feature extraction (frozen Keras backbone).
        2. GLCM + DWT texture feature extraction.
        3. Feature concatenation → [dl_feats | texture_feats].
        4. StandardScaler normalisation.
        5. UMAP transform → 3-D embedding (for visualisation).
        6. LightGBM classification → Mayo Score prediction + probabilities.

    Parameters
    ----------
    rgb_image : np.ndarray, shape (H, W, 3), dtype uint8

    Returns
    -------
    dict
        {
            "mayo_score":      int,          # predicted class (0-3)
            "mayo_label":      str,          # human-readable label
            "probabilities":   list[float],  # softmax-like per-class probs
            "confidence":      float,        # max probability
            "umap_embedding":  list[float],  # [u1, u2, u3] for 3-D plot
            "texture_metrics": dict,         # aggregated texture feature values
            "n_dl_features":   int,
            "n_texture_features": int,
        }
    """
    # ------------------------------------------------------------------
    # Step 1: Deep-learning feature extraction
    # ------------------------------------------------------------------
    feature_extractor, n_dl_features = _get_feature_extractor()
    dl_input  = _preprocess_for_dl(rgb_image)
    dl_feats  = feature_extractor.predict(dl_input, verbose=0)  # (1, n_dl)
    dl_feats  = dl_feats.reshape(1, -1)                         # ensure 2-D

    # ------------------------------------------------------------------
    # Step 2: Texture feature extraction
    # ------------------------------------------------------------------
    texture_feats = extract_glcm_dwt(rgb_image)                 # (n_texture,)
    texture_feats = texture_feats.reshape(1, -1)                # (1, n_texture)

    # ------------------------------------------------------------------
    # Step 3: Concatenate DL + Texture
    # ------------------------------------------------------------------
    combined = np.concatenate([dl_feats, texture_feats], axis=1)  # (1, n_total)

    # ------------------------------------------------------------------
    # Step 4: Scaler normalisation
    # ------------------------------------------------------------------
    scaler         = _get_scaler()
    scaled_combined = scaler.transform(combined)                   # (1, n_total)

    # ------------------------------------------------------------------
    # Step 5: UMAP 3-D embedding (for visualisation)
    # ------------------------------------------------------------------
    umap_reducer  = _get_umap()
    umap_embedding = umap_reducer.transform(scaled_combined)       # (1, 3)
    umap_coords    = umap_embedding[0].tolist()                    # [u1, u2, u3]

    # ------------------------------------------------------------------
    # Step 6: LightGBM classification
    # The LightGBM agent expects the scaled full-dimensional feature vector.
    # (UMAP is a visualisation tool; the GBM operates in the full feature space.)
    # ------------------------------------------------------------------
    booster    = _get_lgb_agent()
    raw_preds  = booster.predict(scaled_combined)                  # (1, n_classes)
    probs      = raw_preds[0].tolist()                             # [p0, p1, p2, p3]
    predicted_class = int(np.argmax(probs))
    confidence      = float(np.max(probs))

    # ------------------------------------------------------------------
    # Texture metric summary (aggregated for display)
    # ------------------------------------------------------------------
    t = texture_feats[0]  # (n_texture,)
    # GLCM layout: 4 props × 2 distances × 4 angles = 32 values
    # DWT layout: 4 sub-bands × 2 stats = 8 values
    texture_summary = {
        "glcm_contrast_mean":     float(t[0:8].mean()),
        "glcm_homogeneity_mean":  float(t[8:16].mean()),
        "glcm_energy_mean":       float(t[16:24].mean()),
        "glcm_correlation_mean":  float(t[24:32].mean()),
        "dwt_ll_energy":          float(t[32]),
        "dwt_ll_variance":        float(t[33]),
        "dwt_lh_energy":          float(t[34]),
        "dwt_lh_variance":        float(t[35]),
        "dwt_hl_energy":          float(t[36]),
        "dwt_hl_variance":        float(t[37]),
        "dwt_hh_energy":          float(t[38]),
        "dwt_hh_variance":        float(t[39]),
    }

    return {
        "mayo_score":        predicted_class,
        "mayo_label":        MAYO_LABELS[predicted_class],
        "probabilities":     probs,
        "confidence":        round(confidence, 4),
        "umap_embedding":    [round(v, 6) for v in umap_coords],
        "texture_metrics":   texture_summary,
        "n_dl_features":     n_dl_features,
        "n_texture_features": N_TEXTURE_FEATURES,
    }


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@bp.route("/health", methods=["GET"])
def health_check():
    """
    Lightweight liveness probe.

    Returns
    -------
    JSON: {"status": "ok"}
    """
    return jsonify({"status": "ok"})


@bp.route("/predict", methods=["POST"])
def predict():
    """
    Primary inference endpoint.

    Accepts a multipart/form-data POST with a single field named 'image'.
    Accepted formats: JPEG, PNG, BMP, TIFF.

    Returns
    -------
    JSON (200) with inference result dict on success.
    JSON (400) if no image is provided or input is invalid.
    JSON (500) if an internal error occurs during inference.
    """
    # Validate request
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. "
                                 "Send a multipart/form-data POST with key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename. Please attach a valid image file."}), 400

    try:
        # Read and preprocess image
        rgb_image = _read_image(file)
    except Exception as exc:
        current_app.logger.error(f"Image read error: {exc}")
        return jsonify({"error": f"Could not read image: {str(exc)}"}), 400

    try:
        # Run hybrid inference pipeline
        result = _run_inference(rgb_image)
    except Exception as exc:
        current_app.logger.error(
            f"Inference error: {exc}\n{traceback.format_exc()}"
        )
        return jsonify({"error": "Internal inference error. See server logs."}), 500

    return jsonify(result), 200


@bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch inference endpoint.

    Accepts a multipart/form-data POST with multiple files, all under the
    key 'images[]'. Processes each image sequentially and returns a list
    of result dicts in the same order.

    Returns
    -------
    JSON (200) with {"results": [...]}.
    JSON (400) if no images are provided.
    """
    files = request.files.getlist("images[]")
    if not files:
        return jsonify({"error": "No images provided. "
                                 "Send files under the key 'images[]'."}), 400

    results = []
    for idx, file in enumerate(files):
        try:
            rgb_image = _read_image(file)
            result    = _run_inference(rgb_image)
            result["filename"] = file.filename
            results.append(result)
        except Exception as exc:
            current_app.logger.error(
                f"Batch inference error on file {idx} ({file.filename}): {exc}"
            )
            results.append(
                {
                    "filename": file.filename,
                    "error":    str(exc),
                }
            )

    return jsonify({"results": results, "count": len(results)}), 200


@bp.route("/texture-only", methods=["POST"])
def texture_only():
    """
    Lightweight texture-only endpoint (no DL model required).

    Useful for quick texture metric inspection without loading the GPU model.
    Accepts a multipart/form-data POST with key 'image'.

    Returns
    -------
    JSON (200) with texture feature summary.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    try:
        rgb_image     = _read_image(file)
        texture_feats = extract_glcm_dwt(rgb_image)
        t             = texture_feats

        texture_summary = {
            "glcm_contrast_mean":    float(t[0:8].mean()),
            "glcm_homogeneity_mean": float(t[8:16].mean()),
            "glcm_energy_mean":      float(t[16:24].mean()),
            "glcm_correlation_mean": float(t[24:32].mean()),
            "dwt_ll_energy":         float(t[32]),
            "dwt_ll_variance":       float(t[33]),
            "dwt_lh_energy":         float(t[34]),
            "dwt_lh_variance":       float(t[35]),
            "dwt_hl_energy":         float(t[36]),
            "dwt_hl_variance":       float(t[37]),
            "dwt_hh_energy":         float(t[38]),
            "dwt_hh_variance":       float(t[39]),
            "raw_vector_length":     int(len(t)),
        }

        return jsonify({"texture_metrics": texture_summary, "filename": file.filename}), 200

    except Exception as exc:
        current_app.logger.error(f"Texture extraction error: {exc}")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Application factory helper
# ---------------------------------------------------------------------------

def create_app(test_config: dict | None = None):
    """
    Flask application factory.

    Parameters
    ----------
    test_config : dict, optional
        Override configuration for testing.

    Returns
    -------
    flask.Flask
    """
    from flask import Flask  # noqa: PLC0415

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-change-in-prod"),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16 MB upload limit
    )

    if test_config is not None:
        app.config.update(test_config)

    # Register inference blueprint
    app.register_blueprint(bp, url_prefix="/api")

    @app.route("/")
    def index():
        """Root endpoint — simple status page."""
        return jsonify(
            {
                "service":  "Colonomind Endoscopic Classifier",
                "version":  "2.0.0",
                "pipeline": "DL + GLCM/DWT Hybrid",
                "endpoints": [
                    "POST /api/predict",
                    "POST /api/predict/batch",
                    "POST /api/texture-only",
                    "GET  /api/health",
                ],
            }
        )

    return app


# ---------------------------------------------------------------------------
# Dev server entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
