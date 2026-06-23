"""
dgx_refit_pipeline.py
---------------------
DGX Server Refit Pipeline — run this ONCE on the DGX server where the
LIMUC dataset is available.

Purpose:
    1. Load every endoscopic image from the LIMUC dataset.
    2. Extract deep-learning (DL) feature vectors from the frozen Keras model
       (best_model.h5) by removing the final classification head.
    3. Extract hybrid texture features (GLCM + DWT) via app/texture_extractor.py.
    4. Concatenate DL + Texture feature vectors.
    5. Fit a new StandardScaler on the concatenated dataset.
    6. Fit a new UMAP (n_components=3) on the scaled dataset.
    7. Persist both artefacts to new_model/scaler_final.pkl and
       new_model/umap_final.pkl, overwriting the previous versions.

IMPORTANT: This script does NOT retrain best_model.h5.  The Keras backbone
is loaded in inference-only mode with weights frozen and the classification
head stripped.

Expected LIMUC dataset layout (configure LIMUC_ROOT below):
    <LIMUC_ROOT>/
      0/   ← Mayo Score 0 images (*.jpg or *.png)
      1/   ← Mayo Score 1
      2/   ← Mayo Score 2
      3/   ← Mayo Score 3

Usage on DGX:
    cd /path/to/Colonomind-Texture-analysis
    python dgx_refit_pipeline.py \
        --limuc_root /data/LIMUC \
        --batch_size 32 \
        --n_jobs -1

Dependencies (install into the DGX environment):
    tensorflow>=2.12  (or torch + timm if the model was converted)
    scikit-learn>=1.3
    umap-learn>=0.5
    Pillow>=9.0
    pywt (PyWavelets)>=1.4
    scikit-image>=0.21
    tqdm
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path bootstrap – allow running from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from app.texture_extractor import extract_glcm_dwt, N_TEXTURE_FEATURES


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

MODEL_PATH   = SCRIPT_DIR / "model-colono" / "models-TryFindingBestModel.h5"
SCALER_PATH  = SCRIPT_DIR / "model-colono" / "models-scaler_agent.pkl"
UMAP_PATH    = SCRIPT_DIR / "model-colono" / "models-umap_model_mixed.pkl"

# Target image size fed to the Keras backbone (must match training resolution)
TARGET_IMAGE_SIZE = (224, 224)

# UMAP hyperparameters
UMAP_N_COMPONENTS  = 3
UMAP_N_NEIGHBORS   = 15
UMAP_MIN_DIST      = 0.1
UMAP_METRIC        = "cosine"
UMAP_RANDOM_STATE  = 42

# Mayo Score class labels
MAYO_CLASSES = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Helper: load Keras model and strip classification head
# ---------------------------------------------------------------------------

def _build_feature_extractor(model_path: Path):
    """
    Load the trained Keras model and return a new model whose output is the
    penultimate feature map (i.e., the layer immediately before the final
    Dense/softmax classification layer).

    The weights are NOT modified; the model backbone is frozen in
    inference-only mode.

    Parameters
    ----------
    model_path : Path
        Absolute path to best_model.h5.

    Returns
    -------
    keras.Model
        Feature-extractor sub-model.
    int
        Number of DL feature dimensions.
    """
    # Lazy import to avoid mandatory TF dependency when this module is imported
    import tensorflow as tf  # noqa: PLC0415

    print(f"[INFO] Loading Keras model from: {model_path}")
    full_model = tf.keras.models.load_model(str(model_path), compile=False)
    full_model.trainable = False  # freeze all weights

    # Identify the output layer – skip the last Dense/Softmax classification layer
    # Strategy: walk backwards until we find the last non-output dense layer.
    feature_layer = None
    for layer in reversed(full_model.layers):
        if isinstance(layer, tf.keras.layers.Dense) and layer.name != full_model.layers[-1].name:
            feature_layer = layer
            break

    if feature_layer is None:
        # Fallback: use the second-to-last layer
        feature_layer = full_model.layers[-2]
        print(
            f"[WARNING] Could not locate penultimate Dense layer automatically. "
            f"Falling back to layer: '{feature_layer.name}' (index -2)."
        )
    else:
        print(f"[INFO] Feature extraction layer: '{feature_layer.name}'")

    # Build a sub-model that outputs at the chosen feature layer
    feature_extractor = tf.keras.Model(
        inputs=full_model.input,
        outputs=feature_layer.output,
        name="feature_extractor",
    )

    n_dl_features = int(feature_extractor.output_shape[-1])
    print(f"[INFO] DL feature dimensionality: {n_dl_features}")
    return feature_extractor, n_dl_features


# ---------------------------------------------------------------------------
# Helper: collect image paths from LIMUC directory tree
# ---------------------------------------------------------------------------

def _collect_limuc_paths(limuc_root: Path):
    """
    Traverse the LIMUC dataset root and return parallel lists of image file
    paths and their corresponding Mayo Score labels.

    Parameters
    ----------
    limuc_root : Path
        Root directory with sub-folders 0/, 1/, 2/, 3/.

    Returns
    -------
    list[Path]  image paths
    list[int]   corresponding Mayo Scores (0, 1, 2, or 3)
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths, labels = [], []

    for mayo_score in MAYO_CLASSES:
        class_dir = limuc_root / str(mayo_score)
        if not class_dir.is_dir():
            print(f"[WARNING] Missing class directory: {class_dir}")
            continue
        for ext in image_extensions:
            for img_path in sorted(class_dir.glob(f"*{ext}")):
                image_paths.append(img_path)
                labels.append(mayo_score)

    print(
        f"[INFO] Collected {len(image_paths)} images across "
        f"{len(set(labels))} Mayo Score classes."
    )
    return image_paths, labels


# ---------------------------------------------------------------------------
# Helper: preprocess a single image for the Keras backbone
# ---------------------------------------------------------------------------

def _preprocess_image_for_dl(img_path: Path) -> np.ndarray:
    """
    Load and preprocess one image for DL feature extraction.

    Returns a float32 array of shape (1, H, W, 3) normalised to [0, 1].
    """
    img = Image.open(img_path).convert("RGB").resize(TARGET_IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def _preprocess_image_for_texture(img_path: Path) -> np.ndarray:
    """
    Load image as uint8 RGB array for texture extraction.

    Returns a uint8 numpy array of shape (H, W, 3).
    """
    img = Image.open(img_path).convert("RGB").resize(TARGET_IMAGE_SIZE)
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_refit_pipeline(
    limuc_root: Path,
    batch_size: int = 32,
    n_jobs: int = -1,
) -> None:
    """
    Execute the full refit pipeline on the LIMUC dataset.

    Steps:
        1. Load feature extractor (frozen Keras backbone).
        2. Iterate the dataset, extracting DL + texture features per image.
        3. Fit StandardScaler on concatenated features.
        4. Fit UMAP on scaled features.
        5. Persist artefacts.

    Parameters
    ----------
    limuc_root : Path
        Root path of the LIMUC dataset.
    batch_size : int
        Number of images per DL inference batch.
    n_jobs : int
        Number of parallel workers for texture extraction.
        -1 uses all available CPU cores.
    """
    # ------------------------------------------------------------------
    # Stage 1: Load feature extractor
    # ------------------------------------------------------------------
    feature_extractor, n_dl_features = _build_feature_extractor(MODEL_PATH)
    n_total_features = n_dl_features + N_TEXTURE_FEATURES
    print(
        f"[INFO] Feature dimensions — DL: {n_dl_features}, "
        f"Texture: {N_TEXTURE_FEATURES}, Total: {n_total_features}"
    )

    # ------------------------------------------------------------------
    # Stage 2: Collect dataset paths
    # ------------------------------------------------------------------
    image_paths, labels = _collect_limuc_paths(limuc_root)
    n_images = len(image_paths)
    if n_images == 0:
        raise RuntimeError(
            f"No images found under {limuc_root}. "
            "Check the LIMUC_ROOT path and directory structure."
        )

    # Pre-allocate feature matrix
    all_features = np.zeros((n_images, n_total_features), dtype=np.float32)

    # ------------------------------------------------------------------
    # Stage 3: Extract features – DL in mini-batches, texture per image
    # ------------------------------------------------------------------
    print(f"\n[INFO] Extracting features for {n_images} images …")

    # --- DL feature extraction (batched for GPU efficiency) ---
    print("[INFO] Phase 1/2: DL feature extraction (batched)")
    dl_features = np.zeros((n_images, n_dl_features), dtype=np.float32)

    for batch_start in tqdm(
        range(0, n_images, batch_size),
        desc="DL batches",
        unit="batch",
    ):
        batch_end  = min(batch_start + batch_size, n_images)
        batch_paths = image_paths[batch_start:batch_end]

        # Stack preprocessed images into a single batch tensor
        batch_arrays = np.concatenate(
            [_preprocess_image_for_dl(p) for p in batch_paths],
            axis=0,
        )  # shape: (B, H, W, 3)

        batch_dl_feats = feature_extractor.predict(batch_arrays, verbose=0)
        dl_features[batch_start:batch_end] = batch_dl_feats.reshape(
            len(batch_paths), -1
        )

    # --- Texture feature extraction (per-image, parallelisable) ---
    print("[INFO] Phase 2/2: Texture feature extraction (per image)")
    texture_features = np.zeros((n_images, N_TEXTURE_FEATURES), dtype=np.float32)

    if n_jobs != 1:
        # Parallel execution via joblib
        from joblib import Parallel, delayed  # noqa: PLC0415

        def _extract_one(img_path: Path) -> np.ndarray:
            arr = _preprocess_image_for_texture(img_path)
            return extract_glcm_dwt(arr)

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_extract_one)(p)
            for p in tqdm(image_paths, desc="Texture", unit="img")
        )
        texture_features = np.array(results, dtype=np.float32)
    else:
        for i, img_path in enumerate(
            tqdm(image_paths, desc="Texture", unit="img")
        ):
            arr = _preprocess_image_for_texture(img_path)
            texture_features[i] = extract_glcm_dwt(arr)

    # --- Concatenate DL + Texture ---
    all_features = np.concatenate([dl_features, texture_features], axis=1)
    labels_array = np.array(labels, dtype=np.int32)

    print(
        f"\n[INFO] Feature matrix shape: {all_features.shape} "
        f"(n_images × n_features)"
    )

    # ------------------------------------------------------------------
    # Stage 4: Fit StandardScaler
    # ------------------------------------------------------------------
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    print("\n[INFO] Fitting StandardScaler …")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)
    print(f"[INFO] Scaler fitted.  Input dim: {scaler.n_features_in_}")

    # ------------------------------------------------------------------
    # Stage 5: Fit UMAP
    # ------------------------------------------------------------------
    import umap as umap_lib  # noqa: PLC0415

    print(f"\n[INFO] Fitting UMAP (n_components={UMAP_N_COMPONENTS}) …")
    umap_reducer = umap_lib.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        n_jobs=n_jobs,
        verbose=True,
    )
    umap_embedding = umap_reducer.fit_transform(scaled_features)
    print(
        f"[INFO] UMAP fitted.  Embedding shape: {umap_embedding.shape}"
    )

    # ------------------------------------------------------------------
    # Stage 6: Persist artefacts
    # ------------------------------------------------------------------
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Saving scaler  → {SCALER_PATH}")
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[INFO] Saving UMAP   → {UMAP_PATH}")
    with open(UMAP_PATH, "wb") as f:
        pickle.dump(umap_reducer, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Optionally save the extracted features and labels for downstream use
    features_cache_path = SCRIPT_DIR / "model-colono" / "cached_features.npz"
    np.savez_compressed(
        features_cache_path,
        features=all_features,
        labels=labels_array,
        dl_features=dl_features,
        texture_features=texture_features,
        umap_embedding=umap_embedding,
    )
    print(f"[INFO] Feature cache → {features_cache_path}")

    print("\n[INFO] Refit pipeline complete.")
    print(
        f"       Artefacts updated:\n"
        f"         Scaler : {SCALER_PATH}\n"
        f"         UMAP   : {UMAP_PATH}\n"
        f"         Cache  : {features_cache_path}"
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DGX Refit Pipeline: fit new StandardScaler + UMAP on the LIMUC "
            "dataset using frozen DL features + GLCM/DWT texture features."
        )
    )
    parser.add_argument(
        "--limuc_root",
        type=Path,
        default=Path("/raid/texture/D13K48009/LIMUC"),
        help="Root directory of the LIMUC dataset (default: /raid/texture/D13K48009/LIMUC).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images per DL inference batch (default: 32).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help=(
            "Number of parallel workers for texture extraction. "
            "-1 uses all CPU cores (default: -1)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.limuc_root.is_dir():
        print(f"[ERROR] LIMUC root directory not found: {args.limuc_root}")
        sys.exit(1)

    run_refit_pipeline(
        limuc_root=args.limuc_root,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
    )
