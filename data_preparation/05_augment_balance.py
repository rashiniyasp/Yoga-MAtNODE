import os
import shutil
import numpy as np
import random
from tqdm import tqdm

# --- CONFIG ---
INPUT_NPY_DIR = r"D:\Yoga-82\Yoga_82_Raw_NPY"
OUTPUT_FINAL_DIR = r"D:\Yoga-82\Yoga_82_Balanced_2026"
TARGET_SAMPLES = 500  # For Training set only

# Mirror pairs for augmentation (Left, Right)
MIRROR_PAIRS = [
    (11,12), (13,14), (15,16), (17,18), (19,20), (21,22),
    (23,24), (25,26), (27,28), (29,30), (31,32), (2,5), (3,6), (7,8)
]

def augment_sample(vector_132):
    """
    Augments a (132,) vector: [Coords(99), Vis(33)]
    """
    coords = vector_132[:99].reshape(33, 3).copy()
    vis = vector_132[99:].copy()
    
    strategy = random.choice(['mirror', 'noise', 'dropout'])
    
    if strategy == 'mirror':
        coords[:, 0] = -coords[:, 0] # Flip X
        for l, r in MIRROR_PAIRS:
            coords[[l, r]] = coords[[r, l]]
            vis[[l, r]] = vis[[r, l]]
            
    elif strategy == 'noise':
        coords += np.random.normal(0, 0.02, coords.shape)
        
    elif strategy == 'dropout':
        idx = np.random.choice(33, random.randint(1, 3), replace=False)
        coords[idx] = 0
        vis[idx] = 0
        
    return np.concatenate([coords.flatten(), vis])

def main():
    if os.path.exists(OUTPUT_FINAL_DIR): shutil.rmtree(OUTPUT_FINAL_DIR)
    
    for split in ['train', 'validation', 'test']:
        src_split = os.path.join(INPUT_NPY_DIR, split)
        dst_split = os.path.join(OUTPUT_FINAL_DIR, split)
        
        if not os.path.exists(src_split): continue
        print(f"Balancing {split}...")
        
        for cls in tqdm(sorted(os.listdir(src_split))):
            s_path = os.path.join(src_split, cls)
            d_path = os.path.join(dst_split, cls)
            os.makedirs(d_path, exist_ok=True)
            
            # Load all originals
            originals = []
            for f in os.listdir(s_path):
                if f.endswith('.npy'):
                    data = np.load(os.path.join(s_path, f))
                    np.save(os.path.join(d_path, f), data) # Save original
                    originals.append(data)
            
            # Augment only Train
            if split == 'train' and len(originals) > 0:
                needed = TARGET_SAMPLES - len(originals)
                for i in range(needed):
                    src_vec = random.choice(originals)
                    aug_vec = augment_sample(src_vec)
                    np.save(os.path.join(d_path, f"aug_{i}.npy"), aug_vec)

    print(f"Done. Final Balanced Dataset: {OUTPUT_FINAL_DIR}")

if __name__ == "__main__":
    main()