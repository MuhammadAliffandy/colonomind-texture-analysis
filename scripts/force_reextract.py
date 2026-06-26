import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Client_Presentation_Analysis.ipynb")
    
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Force LIMUC
            if 'limuc_texture_features.npy' in cell.source:
                cell.source = cell.source.replace('limuc_texture_features.npy', 'limuc_texture_features_3ch_v2.npy')
                cell.source = cell.source.replace('limuc_labels.npy', 'limuc_labels_3ch_v2.npy')
                
            # Force TMC
            if 'tmc_texture_features.npy' in cell.source:
                cell.source = cell.source.replace('tmc_texture_features.npy', 'tmc_texture_features_3ch_v2.npy')
                cell.source = cell.source.replace('tmc_labels.npy', 'tmc_labels_3ch_v2.npy')

    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
        
    print("Notebook modified to use new filenames, forcing re-extraction on the server.")

if __name__ == "__main__":
    main()
