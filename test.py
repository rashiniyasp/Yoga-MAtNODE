import argparse
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

def get_args():
    parser = argparse.ArgumentParser(description="Test Yoga-MAtNODE")
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--model_path', type=str, default='attention_yoga_node_final.pth', help='Path to saved weights')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--use_visibility', action='store_true', help='Use visibility scores')
    return parser.parse_args()

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

def main():
    args = get_args()
    print(f"Loading Test Data from {args.dataset_root}...")
    test_ds = MultiViewYogaDataset(args.dataset_root, 'test', transform=False, use_visibility=args.use_visibility)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Calculate input dim based on settings
    # 212 (Coords+Angles+Bones) or 245 (+Visibility)
    input_dim = 245 if args.use_visibility else 212
    num_classes = len(test_ds.classes)
    
    print(f"Initializing Model (Input Dim: {input_dim}, Num Classes: {num_classes})...")
    model = AttentionYogaNODE(input_dim=input_dim, num_classes=num_classes)
    model.to(DEVICE)
    
    # Load Weights
    print(f"Loading weights from {args.model_path}")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    
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
    print(f"\nFinal Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nGenerating Classification Report...")
    print(classification_report(all_labels, all_preds, target_names=test_ds.classes, digits=4))
    
    plot_confusion_matrix(all_labels, all_preds, test_ds.classes)

if __name__ == "__main__":
    main()