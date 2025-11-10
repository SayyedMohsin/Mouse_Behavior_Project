import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class AdvancedEnsemble:
    def __init__(self, model_configs, device='cuda'):
        self.models = []
        self.device = device
        self.weights = []
        
        for config in model_configs:
            model = self._create_model(config)
            model.load_state_dict(torch.load(config['path']))
            model.to(device)
            model.eval()
            self.models.append(model)
            self.weights.append(config.get('weight', 1.0))
        
        # Normalize weights
        self.weights = np.array(self.weights) / sum(self.weights)
    
    def _create_model(self, config):
        """Create model based on configuration"""
        model_type = config['type']
        
        if model_type == 'transformer':
            from advanced_features.transformer_models import MouseTransformer
            return MouseTransformer(**config.get('params', {}))
        elif model_type == 'vit':
            from advanced_features.transformer_models import VisionTransformerMouse
            return VisionTransformerMouse(**config.get('params', {}))
        else:
            from models import MouseBehaviorClassifier
            return MouseBehaviorClassifier(**config.get('params', {}))
    
    def predict(self, x, method='weighted_average'):
        """Advanced ensemble prediction methods"""
        all_predictions = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                if hasattr(model, 'predict'):
                    output = model.predict(x)
                else:
                    output, _ = model(x)
                
                # Apply softmax if needed
                if output.shape[-1] == 38:  # Assuming 38 classes
                    output = torch.softmax(output, dim=-1)
                
                all_predictions.append(output.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        
        if method == 'weighted_average':
            # Weighted average based on validation performance
            weighted_pred = np.zeros_like(all_predictions[0])
            for i, pred in enumerate(all_predictions):
                weighted_pred += pred * self.weights[i]
            return torch.from_numpy(weighted_pred).to(self.device)
        
        elif method == 'stacking':
            # Use stacking classifier (would need trained meta-learner)
            return self._stacking_predict(all_predictions)
        
        elif method == 'geometric_mean':
            # Geometric mean for probabilities
            log_pred = np.log(all_predictions + 1e-8)
            geometric_mean = np.exp(np.mean(log_pred, axis=0))
            return torch.from_numpy(geometric_mean).to(self.device)
    
    def _stacking_predict(self, predictions):
        """Stacking ensemble prediction"""
        # This would use a trained meta-learner
        # For now, use simple weighted average
        stacked = np.mean(predictions, axis=0)
        return torch.from_numpy(stacked).to(self.device)