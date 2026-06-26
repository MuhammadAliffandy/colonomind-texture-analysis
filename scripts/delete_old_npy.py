import os

def delete_old_npy():
    base_dir = "/Users/aliffandy/Documents/PukulEnam/Colonomind-Texture-analysis"
    data_dir = os.path.join(base_dir, "data")
    
    files_to_delete = [
        os.path.join(data_dir, "limuc_features", "limuc_texture_features.npy"),
        os.path.join(data_dir, "tmc_features", "tmc_texture_features.npy")
    ]
    
    for f in files_to_delete:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted outdated file: {f}")
            
if __name__ == "__main__":
    delete_old_npy()
