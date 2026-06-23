import os
import numpy as np
import pandas as pd
import plotly.express as px

def main():
    print("Mencari file .npy TMC...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "tmc_features")
    report_dir = os.path.join(base_dir, "reports")
    
    tex_path = os.path.join(data_dir, "tmc_texture_features.npy")
    lbl_path = os.path.join(data_dir, "tmc_labels.npy")
    
    if not os.path.exists(tex_path) or not os.path.exists(lbl_path):
        print(f"❌ File {tex_path} atau {lbl_path} tidak ditemukan!")
        print("Pastikan kamu sudah memindahkan hasil download TMC ke folder 'data/tmc_features/'.")
        return

    # Load data
    print("Memuat data TMC...")
    texture_feats = np.load(tex_path)
    labels = np.load(lbl_path)
    
    # Feature names based on the 20-feature architecture
    feat_names = [
        "LL_Mean", "LL_Std", "LL_Var", "LL_Ent", 
        "LH_Mean", "LH_Std", "LH_Var", "LH_Ent",
        "HL_Mean", "HL_Std", "HL_Var", "HL_Ent", 
        "HH_Mean", "HH_Std", "HH_Var", "HH_Ent", "HH_Energy",
        "GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity"
    ]
    
    unique_labels = sorted(list(set(labels)))
    label_map = {l: f"Class {l}" for l in unique_labels}
    # Jika kamu tahu nama kelas aslinya, kamu bisa ubah label_map ini
    
    # Create DataFrame
    df = pd.DataFrame(texture_feats, columns=feat_names)
    df['Label_Name'] = [label_map[l] for l in labels]
    df['Class_Int'] = labels
    
    # Urutkan berdasarkan kelas agar warnanya urut
    df = df.sort_values(by='Class_Int')

    print("Membuat visualisasi interaktif...")
    
    # Gunakan palet warna yang cukup panjang
    color_palette = px.colors.qualitative.Pastel + px.colors.qualitative.Set3
    
    # 1. Scatter Plot (GLCM Contrast vs LL Mean)
    fig1 = px.scatter(
        df, x="GLCM_Contrast", y="LL_Mean", color="Label_Name",
        title="TMC: GLCM Contrast vs DWT LL Mean",
        labels={"GLCM_Contrast": "Kekasaran Tekstur (GLCM Contrast)", "LL_Mean": "Kecerahan Gambar (DWT LL Mean)"},
        color_discrete_sequence=color_palette,
        opacity=0.6
    )
    
    # 2. Box Plot (GLCM Homogeneity)
    fig2 = px.box(
        df, x="Label_Name", y="GLCM_Homogeneity", color="Label_Name",
        title="TMC: Distribusi Keseragaman Tekstur (GLCM Homogeneity)",
        color_discrete_sequence=color_palette
    )

    # 3. Box Plot (DWT HH Variance / Tepi)
    fig3 = px.box(
        df, x="Label_Name", y="HH_Var", color="Label_Name",
        title="TMC: Distribusi Ketajaman Tepi (DWT HH Variance)",
        color_discrete_sequence=color_palette
    )
    
    # Save to HTML
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "Texture_Analysis_Report_TMC.html")
    with open(report_path, "w") as f:
        f.write("<html><head><title>TMC Texture Analysis Report</title>")
        f.write("<style>body{font-family: Arial, sans-serif; background: #0f172a; color: white; padding: 20px;}")
        f.write("h1{text-align: center; color: #38bdf8;} .plot{margin-bottom: 50px; background: white; border-radius: 10px; padding: 10px;}</style></head><body>")
        f.write(f"<h1>Laporan Analisis Tekstur TMC ({len(labels)} Gambar)</h1>")
        
        f.write("<div class='plot'>")
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("</div>")
        
        f.write("<div class='plot'>")
        f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("</div>")
        
        f.write("<div class='plot'>")
        f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("</div>")
        
        f.write("</body></html>")

    print(f"✅ Selesai! Laporan telah disimpan di: {report_path}")

if __name__ == "__main__":
    main()
