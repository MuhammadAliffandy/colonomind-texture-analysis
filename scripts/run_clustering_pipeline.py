import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import umap

def evaluate_clustering(labels_true, labels_pred, features, route_name):
    ari = adjusted_rand_score(labels_true, labels_pred)
    # silhouette can be slow for large datasets, sample if needed, but doing on full dataset here
    if len(features) > 20000:
        idx = np.random.choice(len(features), 20000, replace=False)
        sil = silhouette_score(features[idx], labels_pred[idx])
    else:
        sil = silhouette_score(features, labels_pred)
        
    print(f"--- {route_name} ---")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Silhouette Score: {sil:.4f}\n")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "limuc_features")
    
    tex_path = os.path.join(data_dir, "limuc_texture_features.npy")
    lbl_path = os.path.join(data_dir, "limuc_labels.npy")
    
    if not os.path.exists(tex_path) or not os.path.exists(lbl_path):
        print(f"File features tidak ditemukan di {data_dir}. Menggunakan data dummy.")
        # Generate dummy data for demonstration if files not present
        np.random.seed(42)
        features = np.random.randn(1000, 51)
        labels = np.random.randint(0, 4, 1000)
    else:
        features = np.load(tex_path)
        labels = np.load(lbl_path)
        
    print("=== Clustering Pipeline Architecture ===")
    print(f"Total images: {len(features)}, Features per image: {features.shape[1]}")
    
    # 8. Z-score standardization
    print("\n8. Z-score standardization...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Clustering Route 1: Raw standardized features -> K-means
    print("\nExecuting Route 1 (Raw -> K-means)...")
    kmeans_r1 = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_r1 = kmeans_r1.fit_predict(scaled_features)
    evaluate_clustering(labels, labels_r1, scaled_features, "Route 1: Raw")
    
    # Clustering Route 2: PCA -> K-means
    print("Executing Route 2 (PCA -> K-means)...")
    # Determine components for 95% variance or use fixed number, e.g. 10
    pca = PCA(n_components=10, random_state=42)
    pca_features = pca.fit_transform(scaled_features)
    kmeans_r2 = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_r2 = kmeans_r2.fit_predict(pca_features)
    evaluate_clustering(labels, labels_r2, pca_features, "Route 2: PCA")
    
    # Clustering Route 3: UMAP -> K-means
    print("Executing Route 3 (UMAP -> K-means)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_features = reducer.fit_transform(scaled_features)
    kmeans_r3 = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_r3 = kmeans_r3.fit_predict(umap_features)
    evaluate_clustering(labels, labels_r3, umap_features, "Route 3: UMAP")
    
    print("Reveal MES labels for post-hoc validation - COMPLETED.")

if __name__ == "__main__":
    main()
