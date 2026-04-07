import os

# --- CONFIG ---
DATASET_DIR = r"D:\Yoga-82\dataset_images" 

def rename_folders():
    if not os.path.exists(DATASET_DIR): return
    
    # Check immediate subfolders if structure is flat, or splits if already split
    targets = [DATASET_DIR] 
    # If you run this BEFORE 03, images are just in class folders inside DATASET_DIR
    
    for root in targets:
        for folder in os.listdir(root):
            if "'" in folder:
                old_path = os.path.join(root, folder)
                new_folder = folder.replace("'", "") # e.g. Bharadvaja's -> Bharadvajas
                new_path = os.path.join(root, new_folder)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {folder} -> {new_folder}")
                except Exception as e:
                    print(f"Error renaming {folder}: {e}")

if __name__ == "__main__":
    rename_folders()