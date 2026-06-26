import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Client_Presentation_Analysis.ipynb")
    
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # FIX: Ensure MinMaxScaler is used for bar plot to keep bars pointing up & blue
            if 'import MinMaxScaler' not in cell.source and 'MinMaxScaler' not in cell.source:
                # Add to first cell
                if 'from sklearn.preprocessing import StandardScaler' in cell.source:
                    cell.source = cell.source.replace('from sklearn.preprocessing import StandardScaler', 'from sklearn.preprocessing import StandardScaler, MinMaxScaler')
            
            if 'Evaluasi Clustering (LIMUC)' in cell.source or 'Evaluasi Clustering (TMC)' in cell.source:
                pass # Clustering remains StandardScaler
                
            if 'Visualisasi Feature Importance per MES' in cell.source:
                if 'scaled_limuc' in cell.source:
                    # Switch to MinMaxScaler for plot
                    old_limuc_plot_setup = """variances = np.var(scaled_limuc, axis=0)"""
                    new_limuc_plot_setup = """plot_scaler_limuc = MinMaxScaler(feature_range=(1, 100))
plot_limuc = plot_scaler_limuc.fit_transform(limuc_features)
variances = np.var(plot_limuc, axis=0)"""
                    cell.source = cell.source.replace(old_limuc_plot_setup, new_limuc_plot_setup)
                    
                    cell.source = cell.source.replace('np.mean(scaled_limuc[mes_mask], axis=0)', 'np.mean(plot_limuc[mes_mask], axis=0)')
                    cell.source = cell.source.replace("colors_bar = ['#d62728' if v > 0 else '#1f77b4' for v in vals]", "colors_bar = '#1f77b4'")
                    cell.source = cell.source.replace("color=colors_bar", "color='#1f77b4'")
                    cell.source = cell.source.replace("ax_bar.axhline(0, color='black', linewidth=1)", "")
                    cell.source = cell.source.replace("Standardized Value (Z-Score)", "Relative Feature Strength (0-100)")
                    
                    # Fix circle for LIMUC
                    old_circle = """        offset = 0.1 if y > 0 else -0.1
        ax_bar.add_patch(patches.Ellipse((x, y + offset), width=1.5, height=np.abs(y)*0.4 + 0.1, edgecolor='red', facecolor='none', lw=2))"""
                    new_circle = """        ax_bar.add_patch(patches.Ellipse((x, y + 2), width=1.5, height=y*0.15 + 2, edgecolor='red', facecolor='none', lw=2))"""
                    cell.source = cell.source.replace(old_circle, new_circle)

                if 'scaled_tmc' in cell.source:
                    # Switch to MinMaxScaler for plot
                    old_tmc_plot_setup = """variances_tmc = np.var(scaled_tmc, axis=0)"""
                    new_tmc_plot_setup = """plot_scaler_tmc = MinMaxScaler(feature_range=(1, 100))
    plot_tmc = plot_scaler_tmc.fit_transform(tmc_features)
    variances_tmc = np.var(plot_tmc, axis=0)"""
                    cell.source = cell.source.replace(old_tmc_plot_setup, new_tmc_plot_setup)
                    
                    cell.source = cell.source.replace('np.mean(scaled_tmc[mes_mask_t], axis=0)', 'np.mean(plot_tmc[mes_mask_t], axis=0)')
                    cell.source = cell.source.replace("colors_bar_t = ['#d62728' if v > 0 else '#1f77b4' for v in vals_t]", "colors_bar_t = '#1f77b4'")
                    cell.source = cell.source.replace("color=colors_bar_t", "color='#1f77b4'")
                    cell.source = cell.source.replace("ax_bar.axhline(0, color='black', linewidth=1)", "")
                    cell.source = cell.source.replace("Standardized Value (Z-Score)", "Relative Feature Strength (0-100)")

                    # Fix circle for TMC
                    old_circle_t = """            offset = 0.1 if y > 0 else -0.1
            ax_bar.add_patch(patches.Ellipse((x, y + offset), width=1.5, height=np.abs(y)*0.4 + 0.1, edgecolor='red', facecolor='none', lw=2))"""
                    new_circle_t = """            ax_bar.add_patch(patches.Ellipse((x, y + 2), width=1.5, height=y*0.15 + 2, edgecolor='red', facecolor='none', lw=2))"""
                    cell.source = cell.source.replace(old_circle_t, new_circle_t)
                    
    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
        
    print("Notebook visualization fixed to use MinMaxScaler (0-100) and all blue bars.")

if __name__ == "__main__":
    main()
