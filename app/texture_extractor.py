import os
import cv2
import pywt
import scipy.stats
import numpy as np
try:
    import SimpleITK as sitk
    from radiomics import glcm
except ImportError:
    sitk = None
    glcm = None

# Define Wavelet type
WAVELET = 'db1'

def _smart_preprocess(img):
    if img is None: return None
    h, w = img.shape[:2]
    # Crop if the image is large enough to remove borders
    if h > 450 and w > 550: 
        crop = img[30:430, 200:550]
        if crop.size == 0: crop = img
    else: crop = img
    return cv2.resize(crop, (224, 224))

def _normalize_channel(channel):
    # Normalize to 0-255 for proper PyRadiomics processing and DWT
    return cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def _extract_dwt_17(channel):
    coeffs = pywt.dwt2(channel, WAVELET)
    LL, (LH, HL, HH) = coeffs
    
    def _stats(band):
        flat = np.abs(band.flatten()) + 1e-6
        return [np.mean(band), np.std(band), np.var(band), scipy.stats.entropy(flat)]
    
    feats = []
    for band in [LL, LH, HL, HH]: 
        feats.extend(_stats(band)) 
    feats.append(np.sum(np.square(HH))) # HH Energy
    return feats

def _extract_pyradiomics_glcm(channel):
    if glcm is None or sitk is None:
        # Fallback if pyradiomics not installed
        return [0.0] * 24 # PyRadiomics GLCM usually has 24 features
    
    # Convert to SimpleITK image
    # Add a dummy Z dimension for PyRadiomics (1, H, W)
    arr_3d = np.expand_dims(channel, axis=0)
    image = sitk.GetImageFromArray(arr_3d)
    mask = sitk.GetImageFromArray(np.ones_like(arr_3d, dtype=np.uint8))
    
    settings = {'binWidth': 25, 'force2D': True, 'force2Ddimension': 0}
    glcm_extractor = glcm.RadiomicsGLCM(image, mask, **settings)
    glcm_extractor.enableAllFeatures()
    
    # Execute extraction
    results = glcm_extractor.execute()
    
    # Extract values, sorted by feature name to ensure consistency
    feature_names = sorted(results.keys())
    feats = [results[k].item() for k in feature_names]
    
    return feats

def extract_handcrafted(img):
    """
    Extract features using the 3-channel (Green, Lab-a, Lab-b) architecture:
    DWT-17 + PyRadiomics GLCM per channel.
    """
    # Ensure RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
    # Extract Channels
    # 1. Green
    ch_green = img[:, :, 1]
    
    # 2. Lab-a, Lab-b
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    ch_laba = lab[:, :, 1]
    ch_labb = lab[:, :, 2]
    
    all_features = []
    
    for ch in [ch_green, ch_laba, ch_labb]:
        # Normalize
        ch_norm = _normalize_channel(ch)
        
        # DWT-17
        dwt_feats = _extract_dwt_17(ch_norm)
        
        # PyRadiomics GLCM
        glcm_feats = _extract_pyradiomics_glcm(ch_norm)
        
        all_features.extend(dwt_feats)
        all_features.extend(glcm_feats)
        
    return np.array(all_features, dtype=np.float32)

def extract_glcm_dwt(image_array: np.ndarray) -> np.ndarray:
    """
    Wrapper to match the previous API signature.
    """
    arr = np.asarray(image_array)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    
    processed_img = _smart_preprocess(arr)
    feats = extract_handcrafted(processed_img)
    return feats

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dummy = (rng.random((500, 600, 3)) * 255).astype(np.uint8)
    feats = extract_glcm_dwt(dummy)
    print(f"Output shape: {feats.shape}")
