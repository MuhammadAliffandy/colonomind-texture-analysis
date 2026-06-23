import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def main():
    print("Mencari file .npy...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "limuc_features")
    report_dir = os.path.join(base_dir, "reports")
    
    tex_path = os.path.join(data_dir, "limuc_texture_features.npy")
    lbl_path = os.path.join(data_dir, "limuc_labels.npy")
    
    if not os.path.exists(tex_path) or not os.path.exists(lbl_path):
        print(f"❌ File {tex_path} atau {lbl_path} tidak ditemukan!")
        print("Pastikan kamu sudah melakukan 'git pull' dari server DGX.")
        return

    # Load data
    print("Memuat data...")
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
    
    label_map = {0: "Mayo 0 (Sehat)", 1: "Mayo 1 (Ringan)", 2: "Mayo 2 (Sedang)", 3: "Mayo 3 (Parah)"}
    
    # Create DataFrame
    df = pd.DataFrame(texture_feats, columns=feat_names)
    df['Mayo_Score'] = [label_map[l] for l in labels]
    df['Mayo_Class_Int'] = labels
    
    # Urutkan berdasarkan kelas agar warnanya urut
    df = df.sort_values(by='Mayo_Class_Int')

    print("Membuat visualisasi interaktif...")
    
    # 1. Scatter Plot (GLCM Contrast vs LL Energy)
    fig1 = px.scatter(
        df, x="GLCM_Contrast", y="LL_Mean", color="Mayo_Score",
        title="GLCM Contrast vs DWT LL Mean",
        labels={"GLCM_Contrast": "Kekasaran Tekstur (GLCM Contrast)", "LL_Mean": "Kecerahan Gambar (DWT LL Mean)"},
        color_discrete_sequence=['#10b981', '#fbbf24', '#f97316', '#ef4444'],
        opacity=0.6
    )
    
    # 2. Box Plot (GLCM Homogeneity)
    fig2 = px.box(
        df, x="Mayo_Score", y="GLCM_Homogeneity", color="Mayo_Score",
        title="Distribusi Keseragaman Tekstur (GLCM Homogeneity)",
        color_discrete_sequence=['#10b981', '#fbbf24', '#f97316', '#ef4444']
    )

    # 3. Box Plot (DWT HH Variance / Tepi)
    fig3 = px.box(
        df, x="Mayo_Score", y="HH_Var", color="Mayo_Score",
        title="Distribusi Ketajaman Tepi (DWT HH Variance)",
        color_discrete_sequence=['#10b981', '#fbbf24', '#f97316', '#ef4444']
    )
    
    # Save to HTML
    report_path = os.path.join(report_dir, "Texture_Analysis_Report.html")
    with open(report_path, "w") as f:
        f.write("<html><head><title>Texture Analysis Report</title>")
        f.write("<style>body{font-family: Arial, sans-serif; background: #0f172a; color: white; padding: 20px;}")
        f.write("h1{text-align: center; color: #38bdf8;} .plot{margin-bottom: 50px; background: white; border-radius: 10px; padding: 10px;}</style></head><body>")
        f.write("<h1>Laporan Analisis Tekstur LIMUC (22.552 Gambar)</h1>")
        
        f.write("<div class='plot'>")
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("</div>")
        
        f.write("<div class='plot'>")
        f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
        f.write("</div>")
        
        f.write("<div class='plot'>")
        f.write(fig3.to_html(full_html=False, include_plotlyjs=False))
        f.write("</div>")
        
        f.write("</body></html>")

    print("✅ Selesai! Buka file 'Texture_Analysis_Report.html' di browser (Chrome/Safari) untuk melihat analisis teksturnya.")

if __name__ == "__main__":
    main()
