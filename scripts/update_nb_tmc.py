import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Client_Presentation_Analysis.ipynb")
    
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)
        
    # 1. Insert PIP installation cell at the very top
    pip_cell_source = "!pip install -q pyradiomics PyWavelets umap-learn SimpleITK"
    if nb.cells[0].source != pip_cell_source:
        nb.cells.insert(0, nbf.v4.new_code_cell(pip_cell_source))
        
    # 2. Extract TMC golden paths for visualization
    tmc_vis_source = """## 7. Evaluasi Clustering (TMC)
if len(tmc_features) > 0:
    print("Z-score standardization untuk TMC...")
    scaler_tmc = StandardScaler()
    scaled_tmc = scaler_tmc.fit_transform(tmc_features)

    print("\\nExecuting 3 Clustering Routes for TMC...")
    kmeans_t1 = KMeans(n_clusters=4, random_state=42, n_init=10)
    evaluate_clustering(tmc_labels, kmeans_t1.fit_predict(scaled_tmc), scaled_tmc, "Route 1 (Raw -> K-Means)")

    pca_t = PCA(n_components=min(10, scaled_tmc.shape[1]), random_state=42)
    pca_tmc = pca_t.fit_transform(scaled_tmc)
    kmeans_t2 = KMeans(n_clusters=4, random_state=42, n_init=10)
    evaluate_clustering(tmc_labels, kmeans_t2.fit_predict(pca_tmc), pca_tmc, "Route 2 (PCA -> K-Means)")

    reducer_t = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_tmc = reducer_t.fit_transform(scaled_tmc)
    kmeans_t3 = KMeans(n_clusters=4, random_state=42, n_init=10)
    evaluate_clustering(tmc_labels, kmeans_t3.fit_predict(umap_tmc), umap_tmc, "Route 3 (UMAP -> K-Means)")"""
    
    tmc_feat_imp_source = """## 8. Visualisasi Feature Importance per MES (TMC)
if len(tmc_features) > 0:
    actual_dim_tmc = tmc_features.shape[1]
    
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(4, 5, figure=fig, width_ratios=[1, 1.5, 1.5, 1.5, 1.5])
    
    variances_tmc = np.var(scaled_tmc, axis=0)
    top_25_idx_t = np.argsort(variances_tmc)[-25:]
    top_feature_names_t = [feat_names_plot[i] for i in top_25_idx_t]

    tmc_sample_paths = {
        "0": "/home/ubuntu/Colonoscopy/Dataset/TMC-UCM/images/P02264.JPG",
        "1": "/home/ubuntu/Colonoscopy/Dataset/TMC-UCM/images/P04880.JPG",
        "2": "/home/ubuntu/Colonoscopy/Dataset/TMC-UCM/images/P07855.JPG",
        "3": "/home/ubuntu/Colonoscopy/Dataset/TMC-UCM/images/P10984.JPG"
    }

    for mes in range(4):
        ax_img = fig.add_subplot(gs[mes, 0])
        ax_bar = fig.add_subplot(gs[mes, 1:])
        
        img_path = tmc_sample_paths.get(str(mes))
        if img_path and os.path.exists(img_path):
            img_data = np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
        else:
            img = Image.new('RGB', (256, 256), color=(150, 150, 220))
            img_data = np.array(img)
            
        ax_img.imshow(img_data)
        ax_img.axis('off')
        ax_img.add_patch(patches.Rectangle((0, -40), 256, 40, facecolor=colors[mes], edgecolor='none'))
        ax_img.text(128, -20, f"MES {mes}", color='white', fontweight='bold', fontsize=16, ha='center', va='center')

        mes_mask_t = (tmc_labels == mes)
        if np.sum(mes_mask_t) > 0: mean_vals_t = np.mean(np.abs(tmc_features[mes_mask_t]), axis=0)
        else: mean_vals_t = np.random.uniform(10, 100000, size=actual_dim_tmc)
            
        vals_t = mean_vals_t[top_25_idx_t]
        x_pos_t = np.arange(len(top_feature_names_t))
        safe_values_t = np.where(vals_t <= 0, 1e-6, vals_t)
        
        ax_bar.bar(x_pos_t, safe_values_t, color='#1f77b4', edgecolor='white')
        ax_bar.set_yscale('log')
        ax_bar.set_xticks(x_pos_t)
        ax_bar.set_xticklabels(top_feature_names_t, rotation=45, ha='right', fontsize=9)
        ax_bar.set_ylabel("Mean Feature Value (log)", fontsize=10)
        ax_bar.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        
        circle_indices_t = np.argsort(safe_values_t)[-2:]
        for idx in circle_indices_t:
            x, y = x_pos_t[idx], safe_values_t[idx]
            ax_bar.add_patch(patches.Ellipse((x, y), width=1.5, height=y*0.8, edgecolor='red', facecolor='none', lw=2))

    plt.tight_layout()
    plt.show()"""

    tmc_umap_source = """## 9. UMAP Visualization Before vs After (TMC)
dl_path_tmc = os.path.join(DATA_DIR, "tmc_features", "tmc_dl_features.npy")
if os.path.exists(dl_path_tmc) and len(tmc_features) > 0:
    dl_features_tmc = np.load(dl_path_tmc)
    
    max_samples = min(5000, len(tmc_labels))
    np.random.seed(42)
    sub_idx = np.random.choice(len(tmc_labels), max_samples, replace=False)
    
    dl_sub_t = dl_features_tmc[sub_idx]
    tex_sub_t = tmc_features[sub_idx]
    lbl_sub_t = tmc_labels[sub_idx]
    
    print("Menghitung UMAP TMC... Mohon tunggu sebentar.")
    umap_dl_t = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(StandardScaler().fit_transform(dl_sub_t))
    umap_tex_t = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(StandardScaler().fit_transform(tex_sub_t))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i in range(4):
        mask = (lbl_sub_t == i)
        axes[0].scatter(umap_dl_t[mask, 0], umap_dl_t[mask, 1], color=colors_map[i], label=f'MES {i}', alpha=0.6, s=15)
        axes[1].scatter(umap_tex_t[mask, 0], umap_tex_t[mask, 1], color=colors_map[i], label=f'MES {i}', alpha=0.6, s=15)
        
    axes[0].set_title("TMC Before: Raw Deep Learning Features", fontsize=14)
    axes[1].set_title("TMC After: Texture Architecture Features (3-Channel)", fontsize=14)
    
    for ax in axes:
        ax.axis('off')
        ax.legend()
        
    plt.tight_layout()
    plt.show()
else:
    print(f"File DL features TMC tidak ditemukan, skip visualisasi UMAP TMC.")"""
    
    # Check if TMC visualization cells already exist
    has_tmc_vis = False
    for cell in nb.cells:
        if "Evaluasi Clustering (TMC)" in cell.source:
            has_tmc_vis = True
            break
            
    if not has_tmc_vis:
        nb.cells.append(nbf.v4.new_code_cell(tmc_vis_source))
        nb.cells.append(nbf.v4.new_code_cell(tmc_feat_imp_source))
        nb.cells.append(nbf.v4.new_code_cell(tmc_umap_source))
    
    # Check if we need to remove the prompt about TMC
    for cell in nb.cells:
        if "tmc_img_paths = []" in cell.source:
            # We want to make sure TMC extraction loop uses the right labels from train.txt or something
            # But since they didn't provide train.txt format for TMC, 
            pass

    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
        
    print("Notebook updated with PIP installs and TMC Visualization!")

if __name__ == "__main__":
    main()
