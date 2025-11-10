import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on important keypoints"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv1d(in_channels // 8, in_channels, 1)
        
    def forward(self, x):
        # x shape: (batch, channels, sequence_length)
        attention = torch.mean(x, dim=2, keepdim=True)
        attention = F.relu(self.conv1(attention))
        attention = torch.sigmoid(self.conv2(attention))
        return x * attention

class TemporalAttention(nn.Module):
    """Temporal attention module for focusing on important time steps"""
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch, sequence_length, hidden_size)
        attention_weights = torch.softmax(self.attention(x).squeeze(-1), dim=-1)
        return torch.sum(x * attention_weights.unsqueeze(-1), dim=1), attention_weights

class MouseBehaviorClassifier(nn.Module):
    def __init__(self, num_classes=38, sequence_length=30, feature_dim=128, num_keypoints=10):
        super(MouseBehaviorClassifier, self).__init__()
        
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints
        
        # Keypoint embedding
        self.keypoint_embedding = nn.Linear(3, 32)  # x, y, likelihood for each keypoint
        
        # Spatial processing (per frame)
        self.spatial_attention = SpatialAttention(num_keypoints * 32)
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(num_keypoints * 32, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Temporal processing
        self.lstm = nn.LSTM(256, 512, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.temporal_attention = TemporalAttention(1024)  # 512 * 2 for bidirectional
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, sequence_length, num_keypoints * 3)
        batch_size, seq_len, _ = x.shape
        
        # Reshape for keypoint processing
        x = x.view(batch_size, seq_len, self.num_keypoints, 3)
        
        # Embed each keypoint
        x = self.keypoint_embedding(x)  # (batch, seq_len, num_keypoints, 32)
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, num_keypoints * 32)
        
        # Spatial processing
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        x = self.spatial_attention(x)
        x = self.spatial_conv(x)  # (batch, 256, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, 256)
        
        # Temporal processing
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 1024)
        temporal_features, attention_weights = self.temporal_attention(lstm_out)
        
        # Classification
        output = self.classifier(temporal_features)
        
        return output, attention_weights

class EnsembleModel:
    """Ensemble of multiple models for better performance"""
    def __init__(self, model_paths, device='cuda'):
        self.models = []
        self.device = device
        
        for path in model_paths:
            model = MouseBehaviorClassifier()
            model.load_state_dict(torch.load(path))
            model.to(device)
            model.eval()
            self.models.append(model)
    
    def predict(self, x):
        predictions = []
        with torch.no_grad():
            for model in self.models:
                output, _ = model(x)
                predictions.append(F.softmax(output, dim=1))
        
        # Average predictions
        avg_prediction = torch.mean(torch.stack(predictions), dim=0)
        return avg_prediction