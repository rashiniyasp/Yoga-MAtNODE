import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# --- CONFIG ---
INPUT_IMAGES_DIR = r"D:\Yoga-82\Yoga_82_Split_Images"
OUTPUT_NPY_DIR = r"D:\Yoga-82\Yoga_82_Raw_NPY"

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True, model_complexity=2, 
    enable_segmentation=False, min_detection_confidence=0.5
)

def get_center(landmarks, idx1, idx2):
    return (landmarks[idx1] + landmarks[idx2]) * 0.5

def normalize_and_extract(landmarks_list):
    """
    1. Root Center (Hips) -> 0,0,0
    2. Scale (Torso) -> 1.0
    3. Rotate (Shoulders) -> Align with X-axis
    Returns: [Coords(99), Visibility(33)] (132 dims)
    """
    # Convert to numpy
    xyz = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list])
    vis = np.array([lm.visibility for lm in landmarks_list])
    
    # 1. Center
    hip_center = get_center(xyz, 23, 24) # Left Hip 23, Right Hip 24
    xyz -= hip_center
    
    # 2. Scale
    shoulder_center = get_center(xyz, 11, 12)
    torso_size = np.linalg.norm(shoulder_center)
    if torso_size < 1e-6: return None
    xyz /= torso_size
    
    # 3. Rotate (Align Shoulders to X-axis)
    left, right = xyz[11], xyz[12]
    vec = left - right
    angle = np.arctan2(vec[2], vec[0]) # Angle in X-Z plane
    
    c, s = np.cos(-angle), np.sin(-angle)
    rot_mat = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    xyz = np.dot(xyz, rot_mat.T)
    
    # Return Coords and Visibility only. 
    # Angles/Bones will be calculated dynamically in Dataset.py
    return np.concatenate([xyz.flatten(), vis])

def process_split(split_name):
    src = os.path.join(INPUT_IMAGES_DIR, split_name)
    dst = os.path.join(OUTPUT_NPY_DIR, split_name)
    
    if not os.path.exists(src): return
    
    print(f"Processing {split_name}...")
    classes = sorted(os.listdir(src))
    
    for cls in tqdm(classes):
        cls_src = os.path.join(src, cls)
        cls_dst = os.path.join(dst, cls)
        os.makedirs(cls_dst, exist_ok=True)
        
        for img_name in os.listdir(cls_src):
            img_path = os.path.join(cls_src, img_name)
            
            # Read
            img = cv2.imread(img_path)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract
            res = pose.process(img_rgb)
            if res.pose_world_landmarks:
                features = normalize_and_extract(res.pose_world_landmarks.landmark)
                if features is not None:
                    save_name = os.path.splitext(img_name)[0] + '.npy'
                    np.save(os.path.join(cls_dst, save_name), features)

def main():
    for split in ['train', 'validation', 'test']:
        process_split(split)
    print(f"Extraction Done. Saved to {OUTPUT_NPY_DIR}")

if __name__ == "__main__":
    main()