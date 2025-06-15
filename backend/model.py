"""
PyTorch model definition for the CombinedScorePredictor.
"""
import torch
import torch.nn as nn
from . import config as cfg

class CombinedScorePredictor(nn.Module):
    def __init__(self, n_domains, n_users, domain_emb_dim=16, user_emb_dim=24,
                 title_emb_dim=200, numerical_dim=36, hidden_dim=128, dropout=0.15):
        super(CombinedScorePredictor, self).__init__()

        # Learnable embeddings with MUCH smaller initialization
        self.domain_embedding = nn.Embedding(n_domains, domain_emb_dim)
        self.user_embedding = nn.Embedding(n_users, user_emb_dim)
        
        # FIXED: Much smaller initialization to prevent large negative R² at start
        nn.init.normal_(self.domain_embedding.weight, 0, 0.01)  # Reduced from 0.1 to 0.01
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)    # Reduced from 0.1 to 0.01

        # Calculate total input dimension (full architecture)
        total_input_dim = title_emb_dim + numerical_dim + domain_emb_dim + user_emb_dim
        print(f"🏗️  Model architecture: {title_emb_dim}D title + {numerical_dim}D numerical + {domain_emb_dim}D domain + {user_emb_dim}D user = {total_input_dim}D total")

        # Simpler neural network architecture for stable training
        self.model = nn.Sequential(
            # First layer
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            # Final layer
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Apply He initialization for ReLU layers
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization for ReLU activation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, title_emb, numerical_features, domain_ids, user_ids):
        # Get trainable embeddings for domains and users
        domain_emb = self.domain_embedding(domain_ids)
        user_emb = self.user_embedding(user_ids)

        # FIXED: Remove aggressive scaling that was causing instability
        # Instead use gentle scaling that preserves the small initialization
        
        # Title embeddings: gentle scaling to match numerical scale
        title_scaled = title_emb * 0.5  # Gentle scaling down from ~0.27 std
        
        # Domain/user embeddings: NO scaling to preserve small initialization
        # This prevents the huge negative R² at the start of training
        domain_scaled = domain_emb  # Keep as-is (std ~0.01)
        user_scaled = user_emb      # Keep as-is (std ~0.01)
        
        # Concatenate all features with numerical features dominating initially
        combined = torch.cat([
            title_scaled,           # Gently scaled title embeddings (200D, std ~0.14)
            numerical_features,     # Standardized numerical features (36D, std ~1.0) 
            domain_scaled,          # Small domain embeddings (16D, std ~0.01)
            user_scaled             # Small user embeddings (24D, std ~0.01)
        ], dim=1)                   # Total: 276D - numerical features dominate initially

        # Forward pass through the network
        return self.model(combined).squeeze(1)
    
    def get_embedding_regularization_loss(self):
        """Add L2 regularization for embeddings to prevent overfitting."""
        domain_reg = torch.norm(self.domain_embedding.weight, p=2)
        user_reg = torch.norm(self.user_embedding.weight, p=2)
        return 0.001 * (domain_reg + user_reg)  # Small regularization coefficient