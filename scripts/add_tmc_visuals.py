import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Client_Presentation_Analysis.ipynb")
    
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if "## 10. Advanced Visualizations" in cell.source:
                # Memeriksa apakah TMC plots sudah ada
                if "TMC Scatter & Box Plots" not in cell.source:
                    tmc_vis = """
    # --- VISUALISASI TMC ---
    if len(tmc_features) > 0:
        # 1. SCATTER PLOT TMC: Contrast vs LL Mean
        plt.figure(figsize=(8, 6))
        for mes in range(4):
            mask_t = (tmc_labels == mes)
            x_vals_t = tmc_features[mask_t, idx_contrast]
            y_vals_t = tmc_features[mask_t, idx_ll_mean]
            
            lbl = f"Mayo {mes}"
            if mes == 0: lbl += " (Healthy)"
            elif mes == 1: lbl += " (Mild)"
            elif mes == 2: lbl += " (Moderate)"
            elif mes == 3: lbl += " (Severe)"
                
            plt.scatter(x_vals_t, y_vals_t, c=[colors_map[mes]], label=lbl, alpha=0.7, s=20)
            
        plt.title("TMC: GLCM Contrast vs DWT LL Mean", fontsize=14, fontweight='bold')
        plt.xlabel("Texture Roughness (GLCM Contrast)", fontsize=12)
        plt.ylabel("Image Brightness (DWT LL Mean)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        # 2. BOX PLOT TMC: HH Variance
        plt.figure(figsize=(8, 6))
        df_box_t = pd.DataFrame({
            'DWT HH Variance': tmc_features[:, idx_hh_var],
            'Class': tmc_labels
        })
        
        df_box_t['Class'] = df_box_t['Class'].map(label_map)
        sns.boxplot(x='Class', y='DWT HH Variance', data=df_box_t, palette="husl", showfliers=False)
        plt.title("TMC: Edge Sharpness (DWT HH Variance)", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()"""
                    cell.source = cell.source + "\n" + tmc_vis
                    
    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
        
    print("TMC custom visualizations added!")

if __name__ == "__main__":
    main()
