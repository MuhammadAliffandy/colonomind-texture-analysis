import os
import cv2
import pywt
import scipy.stats
import numpy as np
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

# Define Wavelet type
WAVELET = 'db1'

# We output exactly 20 features
N_TEXTURE_FEATURES = 20

def _smart_preprocess(img):
    if img is None: return None
    h, w = img.shape[:2]
    # Crop if the image is large enough, then resize to 224x224 (model input size)
    if h > 450 and w > 550: 
        crop = img[30:430, 200:550]
        if crop.size == 0: crop = img
    else: crop = img
    return cv2.resize(crop, (224, 224))

def extract_handcrafted(img):
    """
    Extract exactly 20 handcrafted features (17 DWT + 3 GLCM) 
    as defined in the colonomind-rework architecture.
    """
    if len(img.shape) == 3: 
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else: 
        gray = img
    
    # DWT Features (16 stats + 1 HH Energy = 17)
    coeffs = pywt.dwt2(gray, WAVELET)
    LL, (LH, HL, HH) = coeffs
    
    def _stats(band):
        flat = np.abs(band.flatten()) + 1e-6
        return [np.mean(band), np.std(band), np.var(band), scipy.stats.entropy(flat)]
    
    feats = []
    for band in [LL, LH, HL, HH]: 
        feats.extend(_stats(band)) 
    feats.append(np.sum(np.square(HH))) 
    
    # GLCM Features (Contrast, Dissimilarity, Homogeneity = 3)
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(gray_norm, [5], [0, np.pi/4, np.pi/2], 256, symmetric=True, normed=True)
    feats.extend([
        graycoprops(glcm, 'contrast').mean(), 
        graycoprops(glcm, 'dissimilarity').mean(), 
        graycoprops(glcm, 'homogeneity').mean()
    ])
    
    return np.array(feats, dtype=np.float32)

def extract_glcm_dwt(image_array: np.ndarray) -> np.ndarray:
    """
    Wrapper to match the previous API signature while using the new
    20-feature extraction method.
    """
    arr = np.asarray(image_array)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    
    # Apply the same smart crop/resize as the rework architecture
    processed_img = _smart_preprocess(arr)
    feats = extract_handcrafted(processed_img)
    return feats

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dummy = (rng.random((500, 600, 3)) * 255).astype(np.uint8)
    feats = extract_glcm_dwt(dummy)
    print(f"Output shape: {feats.shape}")
    print(f"Features: {feats}")
