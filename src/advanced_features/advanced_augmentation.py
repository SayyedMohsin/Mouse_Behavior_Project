import torch
import numpy as np
import random

class AdvancedMouseAugmentation:
    def __init__(self):
        self.augmentation_methods = []
    
    def temporal_augmentation(self, sequence, labels=None):
        """Advanced temporal augmentations"""
        augmented = sequence.clone()
        
        # Random temporal scaling (speed changes)
        if random.random() < 0.3:
            scale_factor = random.uniform(0.8, 1.2)
            new_length = int(len(sequence) * scale_factor)
            
            if new_length < len(sequence):
                # Speed up - take every nth frame
                step = len(sequence) // new_length
                augmented = sequence[::step][:new_length]
            else:
                # Slow down - interpolate
                indices = np.linspace(0, len(sequence)-1, new_length)
                augmented = torch.stack([sequence[int(i)] for i in indices])
        
        # Random temporal shifting
        if random.random() < 0.4:
            shift = random.randint(-5, 5)
            if shift > 0:
                augmented = torch.cat([augmented[shift:], augmented[:shift]])
            elif shift < 0:
                augmented = torch.cat([augmented[shift:], augmented[:shift]])
        
        # Random frame dropping (simulate occlusions)
        if random.random() < 0.2:
            drop_mask = torch.rand(len(augmented)) > 0.1
            augmented = augmented[drop_mask]
            # Pad to original length
            if len(augmented) < len(sequence):
                padding = augmented[-1:].repeat(len(sequence) - len(augmented), 1)
                augmented = torch.cat([augmented, padding])
        
        return augmented
    
    def spatial_augmentation(self, sequence):
        """Advanced spatial augmentations"""
        augmented = sequence.clone()
        batch_size, seq_len, features = sequence.shape
        
        # Random rotation (mouse orientation)
        if random.random() < 0.3:
            angle = random.uniform(-30, 30) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Rotate coordinates (x, y columns)
            for i in range(0, features, 3):
                if i + 1 < features:  # Check if x, y columns exist
                    x = augmented[:, :, i].clone()
                    y = augmented[:, :, i + 1].clone()
                    
                    # Apply rotation
                    augmented[:, :, i] = x * cos_a - y * sin_a
                    augmented[:, :, i + 1] = x * sin_a + y * cos_a
        
        # Random scaling (size variations)
        if random.random() < 0.2:
            scale = random.uniform(0.9, 1.1)
            for i in range(0, features, 3):
                if i + 1 < features:
                    augmented[:, :, i] *= scale
                    augmented[:, :, i + 1] *= scale
        
        # Random translation (position jitter)
        if random.random() < 0.4:
            tx = random.uniform(-10, 10)
            ty = random.uniform(-10, 10)
            for i in range(0, features, 3):
                if i + 1 < features:
                    augmented[:, :, i] += tx
                    augmented[:, :, i + 1] += ty
        
        # Random noise injection
        if random.random() < 0.3:
            noise = torch.randn_like(augmented) * 0.02
            augmented += noise
        
        return augmented
    
    def behavioral_augmentation(self, sequence, behavior_type):
        """Behavior-specific augmentations"""
        augmented = sequence.clone()
        
        # Aggressive behavior augmentation - more rapid movements
        if 'aggressive' in behavior_type.lower():
            # Increase velocity variance
            movement_indices = [i for i in range(sequence.shape[-1]) if i % 3 != 2]  # Exclude likelihood
            for idx in movement_indices:
                augmented[:, :, idx] *= random.uniform(1.0, 1.5)
        
        # Resting behavior augmentation - smoother movements
        elif 'resting' in behavior_type.lower():
            # Apply temporal smoothing
            kernel_size = 3
            for i in range(augmented.shape[1] - kernel_size + 1):
                window = augmented[:, i:i+kernel_size, :]
                smoothed = window.mean(dim=1, keepdim=True)
                augmented[:, i+1, :] = smoothed.squeeze(1)
        
        return augmented
    
    def __call__(self, sequence, labels=None, behavior_type=None):
        """Apply all augmentations"""
        augmented = sequence.clone()
        
        # Apply temporal augmentations
        augmented = self.temporal_augmentation(augmented, labels)
        
        # Apply spatial augmentations
        augmented = self.spatial_augmentation(augmented)
        
        # Apply behavior-specific augmentations
        if behavior_type:
            augmented = self.behavioral_augmentation(augmented, behavior_type)
        
        return augmented