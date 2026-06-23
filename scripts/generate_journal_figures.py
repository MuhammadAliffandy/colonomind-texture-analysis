import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(dataset_name, base_dir):
    data_dir = os.path.join(base_dir, "data", f"{dataset_name}_features")
    tex_path = os.path.join(data_dir, f"{dataset_name}_texture_features.npy")
    lbl_path = os.path.join(data_dir, f"{dataset_name}_labels.npy")
    
    if not os.path.exists(tex_path) or not os.path.exists(lbl_path):
        print(f"File {tex_path} atau {lbl_path} tidak ditemukan untuk {dataset_name}.")
        return None
    
    texture_feats = np.load(tex_path)
    labels = np.load(lbl_path)
    
    feat_names = [
        "LL_Mean", "LL_Std", "LL_Var", "LL_Ent", 
        "LH_Mean", "LH_Std", "LH_Var", "LH_Ent",
        "HL_Mean", "HL_Std", "HL_Var", "HL_Ent", 
        "HH_Mean", "HH_Std", "HH_Var", "HH_Ent", "HH_Energy",
        "GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity"
    ]
    
    df = pd.DataFrame(texture_feats, columns=feat_names)
    df['Label_Int'] = labels
    return df

def plot_dataset(df, dataset_name, out_dir, label_map, palette):
    print(f"Generating figures for {dataset_name.upper()}...")
    df['Class'] = df['Label_Int'].map(label_map)
    df = df.sort_values(by='Label_Int')
    
    # Set high quality journal style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    # 1. Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="GLCM_Contrast", y="LL_Mean", hue="Class",
        palette=palette, alpha=0.5, s=15, edgecolor=None
    )
    plt.title(f"{dataset_name.upper()}: GLCM Contrast vs DWT LL Mean", fontweight='bold')
    plt.xlabel("Texture Roughness (GLCM Contrast)")
    plt.ylabel("Image Brightness (DWT LL Mean)")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_scatter_contrast_llmean.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_scatter_contrast_llmean.pdf"), bbox_inches='tight')
    plt.close()

    # 2. Box Plot: GLCM Homogeneity
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df, x="Class", y="GLCM_Homogeneity", palette=palette, 
        showfliers=False # hide outliers for cleaner look in journals
    )
    plt.title(f"{dataset_name.upper()}: Texture Homogeneity (GLCM)", fontweight='bold')
    plt.xlabel("Class")
    plt.ylabel("GLCM Homogeneity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_boxplot_homogeneity.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_boxplot_homogeneity.pdf"), bbox_inches='tight')
    plt.close()

    # 3. Box Plot: HH Variance
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df, x="Class", y="HH_Var", palette=palette,
        showfliers=False
    )
    plt.title(f"{dataset_name.upper()}: Edge Sharpness (DWT HH Variance)", fontweight='bold')
    plt.xlabel("Class")
    plt.ylabel("DWT HH Variance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_boxplot_hhvar.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_boxplot_hhvar.pdf"), bbox_inches='tight')
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    # --- LIMUC ---
    df_limuc = load_data("limuc", base_dir)
    if df_limuc is not None:
        label_map_limuc = {0: "Mayo 0 (Healthy)", 1: "Mayo 1 (Mild)", 2: "Mayo 2 (Moderate)", 3: "Mayo 3 (Severe)"}
        palette_limuc = ["#10b981", "#fbbf24", "#f97316", "#ef4444"]
        plot_dataset(df_limuc, "limuc", out_dir, label_map_limuc, palette_limuc)
    
    # --- TMC ---
    df_tmc = load_data("tmc", base_dir)
    if df_tmc is not None:
        unique_labels = sorted(df_tmc['Label_Int'].unique())
        label_map_tmc = {l: f"Class {l}" for l in unique_labels}
        # Get a seaborn color palette for N classes
        palette_tmc = sns.color_palette("husl", len(unique_labels))
        plot_dataset(df_tmc, "tmc", out_dir, label_map_tmc, palette_tmc)

    print(f"✅ Selesai! Semua grafik kualitas jurnal telah disimpan di: {out_dir}")

if __name__ == "__main__":
    main()
