import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Client_Presentation_Analysis.ipynb")
    
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Update paths to match Colab environment
            cell.source = cell.source.replace('limuc_root = "/raid/D13K48009/texture/LIMUC"', 'limuc_root = "/content/Colonomind-Texture-analysis/data" # Atau /content/drive/MyDrive/ jika pakai Drive')
            cell.source = cell.source.replace('data_dir = f"data/{dataset_type}_features"', 'data_dir = f"/content/Colonomind-Texture-analysis/data/{dataset_type}_features"')
            
    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
        
    print("Paths updated for Colab!")

if __name__ == "__main__":
    main()
