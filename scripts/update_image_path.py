import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Client_Presentation_Analysis.ipynb")
    
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)
        
    old_func = """def find_sample_image(limuc_root, mes_label):
    limuc_path = Path(limuc_root)
    if not limuc_path.exists(): return None
    class_mapping = {"0": 0, "1": 1, "2": 2, "3": 3, "Mayo 0": 0, "Mayo 1": 1, "Mayo 2": 2, "Mayo 3": 3}
    for img_path in limuc_path.rglob("*.*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            if img_path.parent.name in class_mapping and class_mapping[img_path.parent.name] == mes_label:
                try: return np.array(Image.open(str(img_path)).convert("RGB").resize((256, 256)))
                except: continue
    return None"""

    new_func = """def find_sample_image(dataset_name, mes_class):
    clean_label = str(int(float(mes_class)))
    img_path = None
    
    if dataset_name == "LIMUC":
        # Menggunakan path Colonoscopy yang spesifik
        limuc_paths = {
            "0": "/home/ubuntu/Colonoscopy/Dataset/LIMUC/patient_based_classified_images/1/Mayo 0/UC_patient_1_16.bmp",
            "1": "/home/ubuntu/Colonoscopy/Dataset/LIMUC/patient_based_classified_images/1/Mayo 1/UC_patient_1_11.bmp",
            "2": "/home/ubuntu/Colonoscopy/Dataset/LIMUC/patient_based_classified_images/1/Mayo 2/UC_patient_1_10.bmp",
            "3": "/home/ubuntu/Colonoscopy/Dataset/LIMUC/patient_based_classified_images/10/Mayo 3/UC_patient_10_27.bmp"
        }
        # Coba replace prefix jika dijalankan di Colab (sesuaikan prefix jika perlu)
        colab_prefix = "/content/drive/MyDrive/"
        
        raw_path = limuc_paths.get(clean_label)
        if raw_path:
            if not os.path.exists(raw_path):
                # Fallback jika di colab
                alt_path = raw_path.replace("/home/ubuntu/", colab_prefix)
                if os.path.exists(alt_path):
                    img_path = alt_path
                else:
                    img_path = raw_path # Biarkan error handling yang urus
            else:
                img_path = raw_path
                
    img_arr = None
    if img_path and os.path.exists(img_path):
        try: img_arr = np.array(Image.open(img_path).convert('RGB').resize((256, 256)))
        except: pass
    return img_arr"""

    for cell in nb.cells:
        if cell.cell_type == 'code':
            if old_func in cell.source:
                cell.source = cell.source.replace(old_func, new_func)
                # Juga perlu update cara pemanggilannya di bawahnya
                cell.source = cell.source.replace('img_data = find_sample_image(limuc_root, mes)', 'img_data = find_sample_image("LIMUC", mes)')
                
    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
        
    print("Notebook image path updated!")

if __name__ == "__main__":
    main()
