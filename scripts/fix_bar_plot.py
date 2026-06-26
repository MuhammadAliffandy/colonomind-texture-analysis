import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Client_Presentation_Analysis.ipynb")
    
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Fix LIMUC Visualization
            if 'limuc_features[mes_mask]' in cell.source and 'scaled_limuc' in cell.source:
                cell.source = cell.source.replace('np.mean(np.abs(limuc_features[mes_mask]), axis=0)', 'np.mean(scaled_limuc[mes_mask], axis=0)')
                cell.source = cell.source.replace('np.random.uniform(10, 100000, size=actual_dim)', 'np.zeros(actual_dim)')
                cell.source = cell.source.replace('safe_values = np.where(vals <= 0, 1e-6, vals)', '')
                cell.source = cell.source.replace("ax_bar.bar(x_pos, safe_values, color='#1f77b4', edgecolor='white')", 
                                                  "colors_bar = ['#d62728' if v > 0 else '#1f77b4' for v in vals]\n    ax_bar.bar(x_pos, vals, color=colors_bar, edgecolor='white')")
                cell.source = cell.source.replace("ax_bar.set_yscale('log')", "ax_bar.axhline(0, color='black', linewidth=1)")
                cell.source = cell.source.replace('Mean Feature Value (log)', 'Standardized Value (Z-Score)')
                
                # Fix red circle logic
                old_circle = """    circle_indices = np.argsort(safe_values)[-2:]
    for idx in circle_indices:
        x, y = x_pos[idx], safe_values[idx]
        ax_bar.add_patch(patches.Ellipse((x, y), width=1.5, height=y*0.8, edgecolor='red', facecolor='none', lw=2))"""
                new_circle = """    circle_indices = np.argsort(np.abs(vals))[-2:]
    for idx in circle_indices:
        x, y = x_pos[idx], vals[idx]
        offset = 0.1 if y > 0 else -0.1
        ax_bar.add_patch(patches.Ellipse((x, y + offset), width=1.5, height=np.abs(y)*0.4 + 0.1, edgecolor='red', facecolor='none', lw=2))"""
                cell.source = cell.source.replace(old_circle, new_circle)

            # Fix TMC Visualization
            if 'tmc_features[mes_mask_t]' in cell.source and 'scaled_tmc' in cell.source:
                cell.source = cell.source.replace('np.mean(np.abs(tmc_features[mes_mask_t]), axis=0)', 'np.mean(scaled_tmc[mes_mask_t], axis=0)')
                cell.source = cell.source.replace('np.random.uniform(10, 100000, size=actual_dim_tmc)', 'np.zeros(actual_dim_tmc)')
                cell.source = cell.source.replace('safe_values_t = np.where(vals_t <= 0, 1e-6, vals_t)', '')
                cell.source = cell.source.replace("ax_bar.bar(x_pos_t, safe_values_t, color='#1f77b4', edgecolor='white')", 
                                                  "colors_bar_t = ['#d62728' if v > 0 else '#1f77b4' for v in vals_t]\n        ax_bar.bar(x_pos_t, vals_t, color=colors_bar_t, edgecolor='white')")
                cell.source = cell.source.replace("ax_bar.set_yscale('log')", "ax_bar.axhline(0, color='black', linewidth=1)")
                cell.source = cell.source.replace('Mean Feature Value (log)', 'Standardized Value (Z-Score)')
                
                # Fix red circle logic for TMC
                old_circle_t = """        circle_indices_t = np.argsort(safe_values_t)[-2:]
        for idx in circle_indices_t:
            x, y = x_pos_t[idx], safe_values_t[idx]
            ax_bar.add_patch(patches.Ellipse((x, y), width=1.5, height=y*0.8, edgecolor='red', facecolor='none', lw=2))"""
                new_circle_t = """        circle_indices_t = np.argsort(np.abs(vals_t))[-2:]
        for idx in circle_indices_t:
            x, y = x_pos_t[idx], vals_t[idx]
            offset = 0.1 if y > 0 else -0.1
            ax_bar.add_patch(patches.Ellipse((x, y + offset), width=1.5, height=np.abs(y)*0.4 + 0.1, edgecolor='red', facecolor='none', lw=2))"""
                cell.source = cell.source.replace(old_circle_t, new_circle_t)
                
    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
        
    print("Notebook visualization fixed to use Z-Score instead of raw log values.")

if __name__ == "__main__":
    main()
