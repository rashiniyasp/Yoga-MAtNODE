import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report


# Import local modules
from model import AttentionYogaNODE
from util import EarlyStopping, plot_history, plot_confusion_matrix
from loss import get_weighted_loss


from dataset import MultiViewYogaDataset

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser(description="Train Yoga-MAtNODE")
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--use_visibility', action='store_true', help='Use visibility scores')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    return parser.parse_args()

def main():
    args = get_args()
    print(f"Initializing Loaders (Visibility={args.use_visibility})...")
    
    # 1. Prepare Datasets
    train_ds = MultiViewYogaDataset(args.dataset_root, 'train', transform=True, use_visibility=args.use_visibility)
    valid_ds = MultiViewYogaDataset(args.dataset_root, 'validation', transform=False, use_visibility=args.use_visibility)
    test_ds = MultiViewYogaDataset(args.dataset_root, 'test', transform=False, use_visibility=args.use_visibility)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)} | Test: {len(test_ds)}")

    # 2. Setup Model
    input_dim = 245 if args.use_visibility else 212
    num_classes = len(train_ds.classes)
    print(f"Detected {num_classes} classes.")
    model = AttentionYogaNODE(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    
    # 3. Setup Loss and Optimizer
    criterion = get_weighted_loss(train_ds, DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopper = EarlyStopping(patience=10, min_delta=0.005)

    # 4. Training Loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}", leave=False)
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Optional: Add noise to features
            inputs = inputs + (torch.randn_like(inputs) * 0.01)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
        
        # Validation
        val_loss, val_acc = validate(model, valid_loader, criterion)
        
        # Store History
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(correct / total)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step()
        print(f"Ep {epoch+1} | Train Acc: {correct/total:.4f} | Val Acc: {val_acc:.4f}")
        
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

    # 5. Save and Evaluate
    torch.save(model.state_dict(), 'attention_yoga_node_final.pth')
    plot_history(history)
    evaluate(model, test_loader)

def validate(model, loader, criterion):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return val_loss / len(loader), correct / total

def evaluate(model, loader):
    print("\nRunning Evaluation on Test Set...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    print(f"Final Test Accuracy: {acc:.4f}")
    
    # Plots
    plot_confusion_matrix(all_labels, all_preds, loader.dataset.classes)
    print(classification_report(all_labels, all_preds, target_names=loader.dataset.classes))

if __name__ == "__main__":
    main()