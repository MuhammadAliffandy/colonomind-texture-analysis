import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEAT_NAMES = [
    "LL_Mean", "LL_Std", "LL_Var", "LL_Ent", 
    "LH_Mean", "LH_Std", "LH_Var", "LH_Ent",
    "HL_Mean", "HL_Std", "HL_Var", "HL_Ent", 
    "HH_Mean", "HH_Std", "HH_Var", "HH_Ent", "HH_Energy",
    "GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity"
]

def generate_rules(dataset_name, texture_path, labels_path, output_md):
    print(f"Generating rule-based thresholds for {dataset_name}...")
    texture_features = np.load(texture_path)
    labels = np.load(labels_path)
    
    # Train RF to get top features
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(texture_features, labels)
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:3] # Top 3 features
    top_features = [FEAT_NAMES[i] for i in top_indices]
    
    df = pd.DataFrame(texture_features, columns=FEAT_NAMES)
    df['Label'] = labels
    
    with open(output_md, 'a') as f:
        f.write(f"## {dataset_name} Rule-Based Thresholds\n\n")
        f.write(f"Fitur paling berpengaruh: **{', '.join(top_features)}**\n\n")
        f.write("| MES Class | Fitur | Rentang Nilai (Q1 - Q3) | Rata-rata (Mean) |\n")
        f.write("|-----------|-------|-------------------------|------------------|\n")
        
        for label in np.unique(labels):
            class_df = df[df['Label'] == label]
            for idx, feat_name in enumerate(top_features):
                q1 = class_df[feat_name].quantile(0.25)
                q3 = class_df[feat_name].quantile(0.75)
                mean_val = class_df[feat_name].mean()
                
                if idx == 0:
                    f.write(f"| **MES {label}** | {feat_name} | {q1:.4f} - {q3:.4f} | {mean_val:.4f} |\n")
                else:
                    f.write(f"| | {feat_name} | {q1:.4f} - {q3:.4f} | {mean_val:.4f} |\n")
        f.write("\n\n")

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    output_dir = os.path.join(base_dir, 'reports')
    output_md = os.path.join(output_dir, 'rule_based_thresholds.md')
    
    if os.path.exists(output_md):
        os.remove(output_md)
        
    with open(output_md, 'w') as f:
        f.write("# Batasan Klasifikasi Tekstur (Rule-Based Thresholds)\n\n")
        f.write("Nilai di bawah ini merupakan estimasi rentang dominan (kuartil ke-25 hingga kuartil ke-75) dari fitur tekstur yang paling berpengaruh untuk setiap kelas Mayo Endoscopic Score (MES).\n\n")
        
    generate_rules(
        'LIMUC',
        os.path.join(base_dir, 'data', 'limuc_features', 'limuc_texture_features.npy'),
        os.path.join(base_dir, 'data', 'limuc_features', 'limuc_labels.npy'),
        output_md
    )
    
    generate_rules(
        'TMC',
        os.path.join(base_dir, 'data', 'tmc_features', 'tmc_texture_features.npy'),
        os.path.join(base_dir, 'data', 'tmc_features', 'tmc_labels.npy'),
        output_md
    )
    print(f"Rule-based thresholds saved to {output_md}")
