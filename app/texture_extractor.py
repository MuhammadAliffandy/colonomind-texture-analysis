"""
app/texture_extractor.py
------------------------
Hybrid texture feature extraction module for the Colonomind pipeline.

Extracts two complementary families of texture descriptors:
  1. GLCM (Gray-Level Co-occurrence Matrix) features via scikit-image
  2. 2-D DWT (Discrete Wavelet Transform) sub-band energy/variance via PyWavelets

The concatenated output is a fixed-length 1-D numpy array that is later
concatenated with the deep-learning feature vector before being passed to
the StandardScaler and UMAP reducer.

Dependencies:
    numpy>=1.23
    scikit-image>=0.21
    PyWavelets>=1.4
"""

import numpy as np
import pywt
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte


# ---------------------------------------------------------------------------
# Public constants – exposed so downstream scripts can query the feature size
# without actually running extraction.
# ---------------------------------------------------------------------------

# GLCM configuration
GLCM_DISTANCES   = [1, 3]                           # pixel distances
GLCM_ANGLES      = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0°, 45°, 90°, 135°
GLCM_PROPERTIES  = ["contrast", "homogeneity", "energy", "correlation"]
GLCM_LEVELS      = 256                              # gray-level quantisation

# DWT configuration
DWT_WAVELET      = "haar"                           # fast, separable, well-studied
DWT_LEVEL        = 1                                # single decomposition level
DWT_SUBBANDS     = ["LL", "LH", "HL", "HH"]        # all 4 sub-bands
DWT_STATS        = ["energy", "variance"]           # per sub-band statistics

# Derived sizes (used externally for validation)
N_GLCM_FEATURES  = len(GLCM_PROPERTIES) * len(GLCM_DISTANCES) * len(GLCM_ANGLES)
N_DWT_FEATURES   = len(DWT_SUBBANDS) * len(DWT_STATS)
N_TEXTURE_FEATURES = N_GLCM_FEATURES + N_DWT_FEATURES   # total feature length


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_grayscale_uint8(image_array: np.ndarray) -> np.ndarray:
    """
    Convert an arbitrary image array to a uint8 grayscale image.

    Parameters
    ----------
    image_array : np.ndarray
        Input image.  Accepted shapes:
          - (H, W)        – already grayscale
          - (H, W, 3)     – RGB or BGR
          - (H, W, 4)     – RGBA (alpha channel is dropped)
        Accepted dtypes: uint8 (0-255), float32/float64 (0.0-1.0 or 0-255).

    Returns
    -------
    np.ndarray  of shape (H, W), dtype=uint8
    """
    arr = np.asarray(image_array)

    # Drop batch dimension if present (e.g. shape (1, H, W, C))
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    # Convert colour to grayscale
    if arr.ndim == 3:
        if arr.shape[2] == 4:          # RGBA – drop alpha
            arr = arr[..., :3]
        arr = rgb2gray(arr)            # returns float64 in [0, 1]

    # Normalise float images to [0, 1] before converting to uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float64)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:          # avoid division by zero for flat images
            arr = (arr - arr_min) / (arr_max - arr_min)
        arr = img_as_ubyte(arr)        # [0, 1] float → [0, 255] uint8

    return arr


# ---------------------------------------------------------------------------
# DWT feature extraction
# ---------------------------------------------------------------------------

def _extract_dwt_features(gray_uint8: np.ndarray) -> np.ndarray:
    """
    Perform a single-level 2-D Haar DWT decomposition and compute per-sub-band
    energy and variance.

    Sub-band order: LL, LH, HL, HH
    Statistics per sub-band: energy (mean of squared coefficients), variance

    Parameters
    ----------
    gray_uint8 : np.ndarray, shape (H, W), dtype uint8

    Returns
    -------
    np.ndarray, shape (N_DWT_FEATURES,)  [8 values total]
    """
    # pywt.dwt2 returns (LL, (LH, HL, HH))
    LL, (LH, HL, HH) = pywt.dwt2(gray_uint8.astype(np.float32), DWT_WAVELET)

    subband_coeffs = {"LL": LL, "LH": LH, "HL": HL, "HH": HH}

    features = []
    for name in DWT_SUBBANDS:
        coeffs = subband_coeffs[name].ravel()
        energy   = float(np.mean(coeffs ** 2))
        variance = float(np.var(coeffs))
        features.extend([energy, variance])

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# GLCM feature extraction
# ---------------------------------------------------------------------------

def _extract_glcm_features(gray_uint8: np.ndarray) -> np.ndarray:
    """
    Compute GLCM-based texture statistics.

    For each combination of (distance × angle), the following properties are
    computed and averaged across distances and angles:
      Contrast, Homogeneity, Energy, Correlation

    Parameters
    ----------
    gray_uint8 : np.ndarray, shape (H, W), dtype uint8

    Returns
    -------
    np.ndarray, shape (N_GLCM_FEATURES,)
        Layout: [contrast_d0a0, contrast_d0a1, …, correlation_d1a3]
        Outer loop: property → distance → angle
    """
    # graycomatrix returns shape (levels, levels, n_distances, n_angles)
    glcm = graycomatrix(
        gray_uint8,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    features = []
    for prop in GLCM_PROPERTIES:
        # graycoprops returns shape (n_distances, n_angles)
        values = graycoprops(glcm, prop)           # shape: (D, A)
        features.extend(values.ravel().tolist())   # row-major: d0a0, d0a1, …

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_glcm_dwt(image_array: np.ndarray) -> np.ndarray:
    """
    Extract a hybrid GLCM + DWT texture feature vector from a single image.

    This is the primary public function consumed by the inference routes and
    the DGX re-fit pipeline.

    Feature layout (fixed-length, N_TEXTURE_FEATURES elements):
    ┌──────────────────────────────────────────────────────────┐
    │  GLCM features  (N_GLCM_FEATURES = 4 props × 2 dist × 4 angles = 32) │
    │  DWT features   (N_DWT_FEATURES  = 4 sub-bands × 2 stats = 8)        │
    └──────────────────────────────────────────────────────────┘
    Total: 40 values

    Parameters
    ----------
    image_array : np.ndarray
        Raw image as loaded by PIL.Image, cv2, or similar.
        Accepted shapes: (H, W), (H, W, 3), (H, W, 4), (1, H, W, 3).
        Accepted dtypes: uint8 or float.

    Returns
    -------
    np.ndarray, shape (N_TEXTURE_FEATURES,), dtype float32
        Flattened 1-D texture feature vector.

    Raises
    ------
    ValueError
        If `image_array` has an unsupported shape or is empty.
    """
    if image_array is None or (hasattr(image_array, "size") and image_array.size == 0):
        raise ValueError("extract_glcm_dwt received an empty image array.")

    arr = np.asarray(image_array)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported image shape: {arr.shape}. "
            "Expected 2-D (H, W) or 3-D (H, W, C)."
        )

    # Convert to uint8 grayscale – both extractors require this
    gray = _to_grayscale_uint8(arr)

    # Extract individual feature families
    glcm_feats = _extract_glcm_features(gray)
    dwt_feats  = _extract_dwt_features(gray)

    # Concatenate into a single feature vector
    texture_features = np.concatenate([glcm_feats, dwt_feats], axis=0)

    assert texture_features.shape[0] == N_TEXTURE_FEATURES, (
        f"Feature dimension mismatch: expected {N_TEXTURE_FEATURES}, "
        f"got {texture_features.shape[0]}."
    )

    return texture_features


# ---------------------------------------------------------------------------
# Quick self-test (run with: python -m app.texture_extractor)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"GLCM feature count : {N_GLCM_FEATURES}")
    print(f"DWT  feature count : {N_DWT_FEATURES}")
    print(f"Total texture feats: {N_TEXTURE_FEATURES}")

    # Smoke-test with a synthetic random RGB image (224×224)
    rng   = np.random.default_rng(42)
    dummy = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
    feats = extract_glcm_dwt(dummy)
    print(f"Output shape       : {feats.shape}")
    print(f"Min / Max / Mean   : {feats.min():.4f} / {feats.max():.4f} / {feats.mean():.4f}")
    print("Self-test PASSED.")
