import torch
import torch.nn as nn
import torch.nn.functional as F

    
class LightWeightAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LightWeightAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded + torch.randn_like(encoded) * 0.1  # Adding noise
        decoded = self.decoder(encoded) 
        
        return decoded