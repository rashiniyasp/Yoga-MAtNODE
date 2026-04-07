import os
import shutil
import random
from tqdm import tqdm

# --- CONFIG ---
SOURCE_DIR = r"D:\Yoga-82\dataset_images"         # Raw downloaded images
TARGET_DIR = r"D:\Yoga-82\Yoga_82_Split_Images"   # New Organized Folder
TRAIN_TXT = r"D:\Yoga-82\yoga_train.txt"
TEST_TXT = r"D:\Yoga-82\yoga_test.txt"

def move_files(file_lines, split_name):
    print(f"Moving {len(file_lines)} images to {split_name}...")
    for line in tqdm(file_lines):
        rel_path = line.strip().split(',')[0]
        # Handle folder rename (apostrophe fix)
        rel_path = rel_path.replace("'", "") 
        
        src = os.path.join(SOURCE_DIR, rel_path)
        dst = os.path.join(TARGET_DIR, split_name, rel_path)
        
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

def main():
    if not os.path.exists(SOURCE_DIR): return print("Source not found")
    
    # 1. Read Lists
    with open(TRAIN_TXT, 'r') as f: train_all = f.readlines()
    with open(TEST_TXT, 'r') as f: test_lines = f.readlines()
    
    # 2. Split Strategy: 
    # Original Test -> Test
    # Original Train -> 80% Train, 20% Validation
    random.seed(42)
    random.shuffle(train_all)
    
    split_idx = int(len(train_all) * 0.8)
    train_lines = train_all[:split_idx]
    val_lines = train_all[split_idx:]
    
    # 3. Execute
    move_files(train_lines, 'train')
    move_files(val_lines, 'validation')
    move_files(test_lines, 'test')
    
    print(f"\nDone. Saved to {TARGET_DIR}")

if __name__ == "__main__":
    main()