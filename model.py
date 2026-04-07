import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """The Dynamics Function (The 'Brain' of the ODE)."""
    def __init__(self, dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Softplus(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, t, x):
        return self.net(x)

class AttentionYogaNODE(nn.Module):
    """Hybrid Model: Transformer Encoder + Neural ODE."""
    def __init__(self, input_dim=212, latent_dim=48, ode_hidden_dim=64, num_classes=82):
        super(AttentionYogaNODE, self).__init__()
        
        # 1. Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2. Neural ODE Block (Geometry Refiner)
        self.ode_func = ODEFunc(latent_dim, ode_hidden_dim)
        self.integration_time = torch.tensor([0.0, 1.0]).float()
        
        # 3. SELF-ATTENTION (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, 
            nhead=4, 
            dim_feedforward=latent_dim*2,
            dropout=0.2, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input x: [Batch, 16_Views, Feat_Dim]
        batch_size, num_views, feat_dim = x.shape
        
        # --- Step A: Encode & ODE (Parallel on all views) ---
        x_flat = x.view(-1, feat_dim) 
        z = self.encoder(x_flat)      
        
        if x.is_cuda: self.integration_time = self.integration_time.cuda()
        
        # Run ODE
        out_ode = odeint(self.ode_func, z, self.integration_time, method='dopri5', rtol=1e-3, atol=1e-3)
        z_refined = out_ode[1] # Take t=1
        
        # --- Step B: Transformer (Global Context) ---
        z_sequence = z_refined.view(batch_size, num_views, -1)
        z_transformed = self.transformer(z_sequence) 
        
        # --- Step C: Aggregation ---
        global_repr = torch.mean(z_transformed, dim=1) 
        
        # --- Step D: Classify ---
        return self.classifier(global_repr)