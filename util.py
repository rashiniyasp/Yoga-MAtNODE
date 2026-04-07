import numpy as np
import torch
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. FEATURE ENGINEERING HELPERS
# ==========================================

def compute_angle_3d(a, b, c):
    """Calculates angle between three points in 3D space."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6: return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle) / 180.0 # Normalize 0-1

def get_pose_features_dynamic(landmarks_xyz):
    """
    Input: (33, 3) array of coordinates
    Output: Combined feature vector [Coords + Angles + Bones]
    """
    # 1. Flatten Coords (99)
    coords = landmarks_xyz.flatten()
    
    # 2. Angles (8)
    mp_map = mp.solutions.pose.PoseLandmark
    angle_triplets = [
        (mp_map.RIGHT_SHOULDER.value, mp_map.RIGHT_ELBOW.value, mp_map.RIGHT_WRIST.value),
        (mp_map.RIGHT_HIP.value, mp_map.RIGHT_SHOULDER.value, mp_map.RIGHT_ELBOW.value),
        (mp_map.LEFT_SHOULDER.value, mp_map.LEFT_ELBOW.value, mp_map.LEFT_WRIST.value),
        (mp_map.LEFT_HIP.value, mp_map.LEFT_SHOULDER.value, mp_map.LEFT_ELBOW.value),
        (mp_map.RIGHT_HIP.value, mp_map.RIGHT_KNEE.value, mp_map.RIGHT_ANKLE.value),
        (mp_map.RIGHT_SHOULDER.value, mp_map.RIGHT_HIP.value, mp_map.RIGHT_KNEE.value),
        (mp_map.LEFT_HIP.value, mp_map.LEFT_KNEE.value, mp_map.LEFT_ANKLE.value),
        (mp_map.LEFT_SHOULDER.value, mp_map.LEFT_HIP.value, mp_map.LEFT_KNEE.value),
    ]
    angles = []
    for a, b, c in angle_triplets:
        ang = compute_angle_3d(landmarks_xyz[a], landmarks_xyz[b], landmarks_xyz[c])
        angles.append(ang)
    angles = np.array(angles)

    # 3. Bones/Vectors (35*3 = 105)
    connections = mp.solutions.pose.POSE_CONNECTIONS
    vectors = []
    for start_idx, end_idx in connections:
        vec = landmarks_xyz[end_idx] - landmarks_xyz[start_idx]
        vectors.append(vec)
    vectors = np.array(vectors).flatten()
    
    return np.concatenate([coords, angles, vectors]) # Size: 212

# ==========================================
# 2. TRAINING HELPERS
# ==========================================

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ==========================================
# 3. VISUALIZATION HELPERS
# ==========================================

def plot_history(history):
    """Plots training and validation accuracy/loss."""
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plots a normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(24, 20)) 
    sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()