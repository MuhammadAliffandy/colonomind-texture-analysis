import nbformat as nbf
import os

def main():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    nb_path = os.path.join(base_dir, "Client_Presentation_Analysis.ipynb")
    
    with open(nb_path, 'r') as f:
        nb = nbf.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Force TMC Re-extraction and Fix Logic
            if 'tmc_texture_features_3ch_v2.npy' in cell.source:
                # Rename to v3 to force extraction
                cell.source = cell.source.replace('tmc_texture_features_3ch_v2.npy', 'tmc_texture_features_3ch_v3.npy')
                cell.source = cell.source.replace('tmc_labels_3ch_v2.npy', 'tmc_labels_3ch_v3.npy')
                
                # Replace the entire old extraction logic with the train.txt reading logic
                old_logic = """    for img_path in Path(TMC_RAW_DIR).rglob("*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.bmp']:
            # Asumsikan format: klasifikasi ada di folder parent atau nama file
            # Fallback: assign label dummy jika tidak ada aturan spesifik.
            label = 0 
            if '1' in img_path.parent.name: label = 1
            elif '2' in img_path.parent.name: label = 2
            elif '3' in img_path.parent.name: label = 3
            
            tmc_img_paths.append(str(img_path))
            tmc_img_labels.append(label)
            
    print(f"Menemukan {len(tmc_img_paths)} gambar TMC. Mulai memproses...")
    
    tmc_feats_list = []
    for i, img_path in enumerate(tqdm(tmc_img_paths, desc="TMC Extraction")):
        try:
            img = Image.open(img_path).convert('RGB')
            img_arr = np.array(img)
            feats = extract_3ch_features(img_arr)
            tmc_feats_list.append(feats)
        except Exception as e:
            # Tetap jaga urutan label jika fail
            tmc_feats_list.append(np.zeros(len(FEATURE_NAMES_3CH)))"""
            
                new_logic = """    # Fungsi pembaca kunci jawaban (label) dari text file
    def parse_tmc_split(txt_file):
        imgs, lbls = [], []
        if os.path.exists(txt_file):
            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        basename = parts[0].split('/')[-1]
                        try:
                            label = int(float(parts[1]))
                        except:
                            continue
                            
                        # Coba cari di folder images, atau augment
                        abs_path = os.path.join(TMC_RAW_DIR, "images", basename)
                        if not os.path.exists(abs_path):
                            abs_path = os.path.join(TMC_RAW_DIR, "augment", basename)
                            
                        if os.path.exists(abs_path):
                            imgs.append(abs_path)
                            lbls.append(label)
        return imgs, lbls

    train_imgs, train_lbls = parse_tmc_split(os.path.join(TMC_RAW_DIR, "train.txt"))
    test_imgs, test_lbls = parse_tmc_split(os.path.join(TMC_RAW_DIR, "test.txt"))
    
    tmc_img_paths = train_imgs + test_imgs
    tmc_img_labels = train_lbls + test_lbls
    
    print(f"Menemukan {len(tmc_img_paths)} gambar TMC berlabel dari train/test.txt. Mulai memproses...")
    
    tmc_feats_list = []
    # Filter array jika ada gambar gagal dibaca
    valid_labels = []
    valid_paths = []
    
    for i, img_path in enumerate(tqdm(tmc_img_paths, desc="TMC Extraction")):
        try:
            img = Image.open(img_path).convert('RGB')
            img_arr = np.array(img)
            feats = extract_3ch_features(img_arr)
            tmc_feats_list.append(feats)
            valid_labels.append(tmc_img_labels[i])
            valid_paths.append(img_path)
        except Exception as e:
            continue
            
    tmc_img_labels = valid_labels
    tmc_img_paths = valid_paths"""
                
                if 'if img_path.suffix.lower() in' in cell.source:
                    cell.source = cell.source.replace(old_logic, new_logic)

    with open(nb_path, 'w') as f:
        nbf.write(nb, f)
        
    print("Notebook updated to parse TMC labels correctly and forced to v3.")

if __name__ == "__main__":
    main()
