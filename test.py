import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# --- IMPORTS ---
from model import AttentionYogaNODE
from dataset import MultiViewYogaDataset

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_ROOT = r"D:\Yoga-82\Yoga_82_Balanced_2026" # Path to processed data
MODEL_PATH = "attention_yoga_node_final.pth"       # Path to saved weights
BATCH_SIZE = 256

# IMPORTANT: Must match the setting used during training!
USE_VISIBILITY = False 

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm_norm, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Confusion matrix saved to 'confusion_matrix.png'")
    plt.show()

def main():
    print(f"Loading Test Data from {DATASET_ROOT}...")
    test_ds = MultiViewYogaDataset(DATASET_ROOT, 'test', transform=False, use_visibility=USE_VISIBILITY)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Calculate input dim based on settings
    # 212 (Coords+Angles+Bones) or 245 (+Visibility)
    input_dim = 245 if USE_VISIBILITY else 212
    
    print(f"Initializing Model (Input Dim: {input_dim})...")
    model = AttentionYogaNODE(input_dim=input_dim, num_classes=len(test_ds.classes))
    model.to(DEVICE)
    
    # Load Weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running Inference...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Final Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nGenerating Classification Report...")
    print(classification_report(all_labels, all_preds, target_names=test_ds.classes, digits=4))
    
    plot_confusion_matrix(all_labels, all_preds, test_ds.classes)

if __name__ == "__main__":
    main()