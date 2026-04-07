import torch
import torch.nn as nn

def get_weighted_loss(train_dataset, device, label_smoothing=0.2):
    """
    Calculates class weights based on dataset counts and returns CrossEntropyLoss.
    """
    counts = [train_dataset.label_counts[i] for i in range(len(train_dataset.classes))]
    
    # Avoid division by zero
    counts = [c if c > 0 else 1 for c in counts] 
    
    # Calculate inverse weights
    weights = sum(counts) / (len(counts) * torch.FloatTensor(counts))
    weights = weights.to(device)
    
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)