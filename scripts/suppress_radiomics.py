import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Client_Presentation_Analysis.ipynb")
    
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if "import warnings" in cell.source:
                # Add logging suppression for radiomics
                if "import logging" not in cell.source:
                    new_warnings = """import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("radiomics").setLevel(logging.ERROR)"""
                    cell.source = cell.source.replace("import warnings\nwarnings.filterwarnings('ignore')", new_warnings)

    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
        
    print("Notebook modified to suppress PyRadiomics warnings.")

if __name__ == "__main__":
    main()
