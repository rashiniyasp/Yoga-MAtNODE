import os
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import Counter

# Import the feature extraction logic from your util file
from util import get_pose_features_dynamic

class MultiViewYogaDataset(Dataset):
    """
    PyTorch Dataset for Yoga-82 3D Skeletons.
    
    This dataset:
    1. Loads pre-processed .npy files (containing 3D coords & visibility).
    2. Dynamically generates 16 multi-view rotations on-the-fly.
    3. Re-calculates geometric features (angles/bones) for each view.
    
    Args:
        root_dir (str): Path to the dataset root (e.g., 'Yoga_82_Balanced_2026').
        split (str): 'train', 'validation', or 'test'.
        transform (bool): If True, applies random noise augmentation (usually for training).
        use_visibility (bool): If True, appends visibility scores to the feature vector.
    """
    def __init__(self, root_dir, split='train', transform=False, use_visibility=False):
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.use_visibility = use_visibility
        
        # 1. Validation: Check if directory exists
        if not os.path.exists(self.split_dir):
            # Fallback for 'validation' vs 'valid' naming differences between datasets
            if split == 'validation' and os.path.exists(os.path.join(root_dir, 'valid')):
                self.split_dir = os.path.join(root_dir, 'valid')
            elif split == 'valid' and os.path.exists(os.path.join(root_dir, 'validation')):
                self.split_dir = os.path.join(root_dir, 'validation')
            else:
                raise ValueError(f"Directory not found: {self.split_dir}")

        # 2. Auto-detect Classes
        # We sort to ensure consistent index mapping (0, 1, 2...) across different machines
        self.classes = sorted([d for d in os.listdir(self.split_dir) 
                               if os.path.isdir(os.path.join(self.split_dir, d))])
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 3. Index Files
        # We store paths in a list instead of loading everything to RAM immediately.
        # This is standard practice for scalable code.
        self.samples = [] 
        self.label_counts = Counter() # Used by loss.py for weighting

        print(f"Indexing {split} set...")
        for cls_name in self.classes:
            cls_path = os.path.join(self.split_dir, cls_name)
            class_idx = self.class_to_idx[cls_name]
            
            # List all .npy files
            files = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
            
            for f in files:
                file_path = os.path.join(cls_path, f)
                self.samples.append((file_path, class_idx))
                self.label_counts[class_idx] += 1
                
        print(f"--> Found {len(self.samples)} samples across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def _rotate_and_extract(self, skeleton_xyz, visibility_arr):
        """
        Generates 16 rotated views of the skeleton and extracts features for each.
        """
        views = []
        # Generate 16 angles from -180 to 180 degrees
        angles_deg = np.linspace(-180, 180, 16)
        
        for deg in angles_deg:
            theta = np.deg2rad(deg)
            c, s = np.cos(theta), np.sin(theta)
            
            # Rotation Matrix (Y-axis rotation)
            rot_mat = np.array([
                [c,  0, s],
                [0,  1, 0],
                [-s, 0, c]
            ])
            
            # 1. Rotate the coordinates
            # Shape: (33, 3)
            rotated_xyz = np.dot(skeleton_xyz, rot_mat)
            
            # 2. Extract Geometric Features (Coords + Angles + Bones)
            # This function comes from util.py
            # Returns size 212
            feats = get_pose_features_dynamic(rotated_xyz)
            
            # 3. Append Visibility (Optional)
            if self.use_visibility:
                # Returns size 212 + 33 = 245
                feats = np.concatenate([feats, visibility_arr])
            
            views.append(feats)
            
        return np.array(views, dtype=np.float32)

    def __getitem__(self, idx):
        # 1. Retrieve file path and label
        file_path, label = self.samples[idx]
        
        # 2. Load the raw data
        raw_data = np.load(file_path)
        
        # 3. Parse Components
        if len(raw_data.shape) == 1 and raw_data.shape[0] == 245:
            # Format: [Coords(99) | Angles(8) | Bones(105) | Vis(33)]
            coords = raw_data[:99].reshape(33, 3)
            visibility = raw_data[-33:]
        elif len(raw_data.shape) == 2 and raw_data.shape == (33, 4):
            # Format: (33, 4) - Coords(x,y,z) + Vis
            coords = raw_data[:, :3]
            visibility = raw_data[:, 3]
        elif len(raw_data.shape) == 2 and raw_data.shape == (33, 3):
            # Format: (33, 3) - Coords(x,y,z) only
            coords = raw_data
            visibility = np.ones(33) # Dummy visibility
        else:
            # Fallback if it's some other flat format but starts with coords
            coords = raw_data[:99].reshape(33, 3)
            visibility = raw_data[-33:] if len(raw_data) >= 132 else np.ones(33) 
        
        # 4. Runtime Augmentation (Training Only)
        # Adds slight noise to coordinates before rotation
        if self.transform:
            noise = np.random.normal(0, 0.005, coords.shape)
            coords = coords + noise
            
        # 5. Multi-View Generation
        # Returns Tensor of shape [16, Feature_Dim]
        multi_view_features = self._rotate_and_extract(coords, visibility)
        
        return torch.from_numpy(multi_view_features), torch.tensor(label, dtype=torch.long)