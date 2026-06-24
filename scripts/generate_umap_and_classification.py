import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

FEAT_NAMES = [
    "LL_Mean", "LL_Std", "LL_Var", "LL_Ent", 
    "LH_Mean", "LH_Std", "LH_Var", "LH_Ent",
    "HL_Mean", "HL_Std", "HL_Var", "HL_Ent", 
    "HH_Mean", "HH_Std", "HH_Var", "HH_Ent", "HH_Energy",
    "GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity"
]

def analyze_dataset(dataset_name, dl_path, texture_path, labels_path, output_dir):
    print(f"\n--- Analyzing {dataset_name} ---")
    
    # Load data
    dl_features = np.load(dl_path)
    texture_features = np.load(texture_path)
    labels = np.load(labels_path)
    
    # Subsample for UMAP if large
    subsample_idx = np.arange(dl_features.shape[0])
    if len(subsample_idx) > 10000:
        np.random.seed(42)
        subsample_idx = np.random.choice(subsample_idx, 10000, replace=False)
    
    dl_sub = dl_features[subsample_idx]
    texture_sub = texture_features[subsample_idx]
    labels_sub = labels[subsample_idx]
    
    # Scale features for UMAP to avoid snake-like manifolds
    scaler_dl = StandardScaler()
    dl_sub_scaled = scaler_dl.fit_transform(dl_sub)
    
    scaler_tex = StandardScaler()
    texture_sub_scaled = scaler_tex.fit_transform(texture_sub)
    
    # 1. UMAP Projection
    print("Computing UMAP for DL (Raw) Features...")
    reducer_dl = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    umap_dl = reducer_dl.fit_transform(dl_sub_scaled)
    
    print("Computing UMAP for Texture Features...")
    reducer_texture = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    umap_texture = reducer_texture.fit_transform(texture_sub_scaled)
    
    # Plotting UMAP
    unique_labels = np.unique(labels_sub)
    colors = sns.color_palette("husl", len(unique_labels))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for idx, label in enumerate(unique_labels):
        mask = (labels_sub == label)
        axes[0].scatter(umap_dl[mask, 0], umap_dl[mask, 1], color=colors[idx], label=f'MES {label}', alpha=0.6, s=10)
        axes[1].scatter(umap_texture[mask, 0], umap_texture[mask, 1], color=colors[idx], label=f'MES {label}', alpha=0.6, s=10)
        
    axes[0].set_title(f'{dataset_name} UMAP: Raw Deep Learning Features (Before)', fontsize=14)
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].legend(title='Class')
    
    axes[1].set_title(f'{dataset_name} UMAP: Texture Features (After)', fontsize=14)
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].legend(title='Class')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'figures', f'{dataset_name.lower()}_umap_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved UMAP plot to {plot_path}")
    plt.close()
    
    # 2. Classification & Feature Importance
    print("Training Classification Models...")
    # Texture features model
    X_train_tex, X_test_tex, y_train, y_test = train_test_split(texture_features, labels, test_size=0.2, random_state=42, stratify=labels)
    rf_tex = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_tex.fit(X_train_tex, y_train)
    preds_tex = rf_tex.predict(X_test_tex)
    acc_tex = accuracy_score(y_test, preds_tex)
    report_tex = classification_report(y_test, preds_tex)
    
    # Feature Importance Plot
    importances = rf_tex.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title(f"Feature Importances for {dataset_name} (Texture Analysis)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [FEAT_NAMES[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    fi_plot_path = os.path.join(output_dir, 'figures', f'{dataset_name.lower()}_feature_importance.png')
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'classification_metrics.txt')
    with open(metrics_path, 'a') as f:
        f.write(f"========================================\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"========================================\n")
        f.write(f"[Texture Features (After GLCM & DWT)]\n")
        f.write(f"Accuracy: {acc_tex:.4f}\n")
        f.write(f"Classification Report:\n{report_tex}\n\n")

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    output_dir = os.path.join(base_dir, 'reports')
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    metrics_path = os.path.join(output_dir, 'classification_metrics.txt')
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
        
    analyze_dataset(
        'LIMUC',
        os.path.join(base_dir, 'data', 'limuc_features', 'limuc_dl_features.npy'),
        os.path.join(base_dir, 'data', 'limuc_features', 'limuc_texture_features.npy'),
        os.path.join(base_dir, 'data', 'limuc_features', 'limuc_labels.npy'),
        output_dir
    )
    analyze_dataset(
        'TMC',
        os.path.join(base_dir, 'data', 'tmc_features', 'tmc_dl_features.npy'),
        os.path.join(base_dir, 'data', 'tmc_features', 'tmc_texture_features.npy'),
        os.path.join(base_dir, 'data', 'tmc_features', 'tmc_labels.npy'),
        output_dir
    )
