import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Client_Presentation_Analysis.ipynb")
    
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)
        
    # --- CELL: Visualizations (Scatter & Box Plot) ---
    vis_source = """## 10. Advanced Visualizations (LIMUC Scatter & Box Plots)
import pandas as pd

if len(limuc_features) > 0 and 'Green_GLCM_Contrast' in FEATURE_NAMES_3CH:
    # Get Indices for the requested features
    idx_contrast = FEATURE_NAMES_3CH.index('Green_GLCM_Contrast')
    idx_ll_mean = FEATURE_NAMES_3CH.index('Green_LL_Mean')
    idx_hh_var = FEATURE_NAMES_3CH.index('Green_HH_Var')
    
    # 1. SCATTER PLOT: Contrast vs LL Mean
    plt.figure(figsize=(8, 6))
    colors_map = sns.color_palette("husl", 4)
    for mes in range(4):
        mask = (limuc_labels == mes)
        # Using raw features for interpretation (as in the client's reference)
        x_vals = limuc_features[mask, idx_contrast]
        y_vals = limuc_features[mask, idx_ll_mean]
        
        # Label mapping
        lbl = f"Mayo {mes}"
        if mes == 0: lbl += " (Healthy)"
        elif mes == 1: lbl += " (Mild)"
        elif mes == 2: lbl += " (Moderate)"
        elif mes == 3: lbl += " (Severe)"
            
        plt.scatter(x_vals, y_vals, c=[colors_map[mes]], label=lbl, alpha=0.7, s=20)
        
    plt.title("LIMUC: GLCM Contrast vs DWT LL Mean", fontsize=14, fontweight='bold')
    plt.xlabel("Texture Roughness (GLCM Contrast)", fontsize=12)
    plt.ylabel("Image Brightness (DWT LL Mean)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 2. BOX PLOT: HH Variance (Edge Sharpness)
    plt.figure(figsize=(8, 6))
    
    df_box = pd.DataFrame({
        'DWT HH Variance': limuc_features[:, idx_hh_var],
        'Class': limuc_labels
    })
    
    label_map = {0: "Mayo 0 (Healthy)", 1: "Mayo 1 (Mild)", 2: "Mayo 2 (Moderate)", 3: "Mayo 3 (Severe)"}
    df_box['Class'] = df_box['Class'].map(label_map)
    
    sns.boxplot(x='Class', y='DWT HH Variance', data=df_box, palette="husl", showfliers=False)
    plt.title("LIMUC: Edge Sharpness (DWT HH Variance)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()"""
    nb.cells.append(nbf.v4.new_code_cell(vis_source))

    # --- CELL: Rule-Based Thresholds Table ---
    table_source = """## 11. Rule-Based Thresholds (LIMUC & TMC)
from IPython.display import display, HTML

def generate_threshold_table(features, labels, target_feats, dataset_name):
    if len(features) == 0: return None
    
    # Dapatkan index fitur yang dicari
    indices = []
    headers = []
    for f in target_feats:
        if f in FEATURE_NAMES_3CH:
            indices.append(FEATURE_NAMES_3CH.index(f))
            headers.append(f"{f} Range\\n(Avg)")
        else:
            indices.append(-1)
            headers.append(f)
            
    table_data = []
    for mes in range(4):
        mask = (labels == mes)
        if np.sum(mask) == 0: continue
            
        row = [f"MES {mes}"]
        for idx in indices:
            if idx != -1:
                vals = features[mask, idx]
                q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
                avg = np.mean(vals)
                row.append(f"{q1:.4f} - {q3:.4f}\\n({avg:.4f})")
            else:
                row.append("N/A")
        table_data.append(row)
        
    df = pd.DataFrame(table_data, columns=["MES Level"] + headers)
    
    # Render HTML rapi
    html = f"<h3>{dataset_name} Rule-Based Thresholds</h3>"
    # Convert newlines to <br> for HTML rendering
    df_html = df.to_html(index=False, escape=False).replace('\\n', '<br>')
    html += df_html
    return html

# 1. LIMUC Table
limuc_html = generate_threshold_table(
    limuc_features, 
    limuc_labels, 
    ['Green_HH_Ent', 'Green_HL_Mean', 'Green_LH_Mean'], 
    "LIMUC"
)

# 2. TMC Table
tmc_html = generate_threshold_table(
    tmc_features, 
    tmc_labels, 
    ['Green_HL_Ent', 'Green_LH_Ent', 'Green_GLCM_Dissimilarity'], 
    "TMC"
)

# Tampilkan
final_html = ""
if limuc_html: final_html += limuc_html + "<br><br>"
if tmc_html: final_html += tmc_html

if final_html:
    display(HTML(final_html))
else:
    print("Fitur tidak tersedia untuk mencetak tabel thresholds.")"""
    nb.cells.append(nbf.v4.new_code_cell(table_source))

    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
        
    print("Advanced visualizations and Rule-based tables added to the notebook.")

if __name__ == "__main__":
    main()
