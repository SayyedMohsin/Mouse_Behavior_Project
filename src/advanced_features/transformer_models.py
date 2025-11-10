import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MouseTransformer(nn.Module):
    def __init__(self, num_classes=38, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(MouseTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(30 * 3, d_model)  # 10 keypoints * 3 values
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output

class VisionTransformerMouse(nn.Module):
    """Vision Transformer for mouse pose sequences"""
    def __init__(self, num_classes=38, seq_length=30, patch_size=5, d_model=256):
        super(VisionTransformerMouse, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = (seq_length // patch_size)
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_size * 30, d_model)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))
        
        # Transformer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, 6)
        
        # Classifier
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Create patches
        patches = x.unfold(1, self.patch_size, self.patch_size)  # (batch, num_patches, features, patch_size)
        patches = patches.contiguous().view(batch_size, self.num_patches, -1)
        
        # Embed patches
        patch_embeddings = self.patch_embed(patches)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)
        
        # Add position embedding
        embeddings += self.pos_embed
        
        # Transformer
        transformer_out = self.transformer(embeddings)
        
        # Use class token for classification
        cls_output = transformer_out[:, 0]
        
        return self.classifier(cls_output)