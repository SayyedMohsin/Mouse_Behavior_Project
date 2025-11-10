import numpy as np
import pandas as pd
from scipy import stats
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
    
    def extract_comprehensive_features(self, sequence_data):
        """Extract comprehensive features from mouse tracking sequence"""
        features = []
        
        # Basic position features
        basic_features = self._extract_basic_features(sequence_data)
        features.extend(basic_features)
        
        # Movement features
        movement_features = self._extract_movement_features(sequence_data)
        features.extend(movement_features)
        
        # Posture features
        posture_features = self._extract_posture_features(sequence_data)
        features.extend(posture_features)
        
        # Social features (if multiple mice)
        social_features = self._extract_social_features(sequence_data)
        features.extend(social_features)
        
        # Statistical features
        statistical_features = self._extract_statistical_features(sequence_data)
        features.extend(statistical_features)
        
        return np.array(features)
    
    def _extract_basic_features(self, data):
        """Extract basic position and likelihood features"""
        features = []
        
        # Center of mass position
        com_x = data['center_back_x'].mean()
        com_y = data['center_back_y'].mean()
        features.extend([com_x, com_y])
        
        # Head position
        head_x = data['nose_x'].mean()
        head_y = data['nose_y'].mean()
        features.extend([head_x, head_y])
        
        # Average likelihood of keypoints
        avg_likelihood = data[[col for col in data.columns if 'likelihood' in col]].mean().mean()
        features.append(avg_likelihood)
        
        self.feature_names.extend(['com_x', 'com_y', 'head_x', 'head_y', 'avg_likelihood'])
        
        return features
    
    def _extract_movement_features(self, data):
        """Extract movement-related features"""
        features = []
        
        # Velocity features
        nose_velocities = self._calculate_velocity(data['nose_x'], data['nose_y'])
        com_velocities = self._calculate_velocity(data['center_back_x'], data['center_back_y'])
        
        features.extend([
            np.mean(nose_velocities),  # Average speed
            np.std(nose_velocities),   # Speed variability
            np.max(nose_velocities),   # Maximum speed
            np.mean(com_velocities),
            np.std(com_velocities)
        ])
        
        # Acceleration
        if len(nose_velocities) > 1:
            acceleration = np.diff(nose_velocities)
            features.extend([np.mean(acceleration), np.std(acceleration)])
        else:
            features.extend([0, 0])
        
        self.feature_names.extend([
            'avg_nose_speed', 'std_nose_speed', 'max_nose_speed',
            'avg_com_speed', 'std_com_speed', 'avg_acceleration', 'std_acceleration'
        ])
        
        return features
    
    def _extract_posture_features(self, data):
        """Extract posture and body configuration features"""
        features = []
        
        # Body length (nose to tail base)
        body_lengths = np.sqrt(
            (data['nose_x'] - data['tail_base_x'])**2 + 
            (data['nose_y'] - data['tail_base_y'])**2
        )
        features.extend([np.mean(body_lengths), np.std(body_lengths)])
        
        # Head elevation
        head_elevation = data['nose_y'] - data['neck_y']
        features.extend([np.mean(head_elevation), np.std(head_elevation)])
        
        # Body orientation
        orientations = np.arctan2(
            data['tail_base_y'] - data['nose_y'],
            data['tail_base_x'] - data['nose_x']
        )
        features.extend([np.mean(orientations), np.std(orientations)])
        
        self.feature_names.extend([
            'avg_body_length', 'std_body_length',
            'avg_head_elevation', 'std_head_elevation',
            'avg_orientation', 'std_orientation'
        ])
        
        return features
    
    def _extract_social_features(self, data):
        """Extract social interaction features (placeholder for multi-mouse data)"""
        features = []
        
        # These would be implemented based on multiple mouse tracking
        # For now, return zeros
        social_feats = [0] * 10  # Placeholder
        features.extend(social_feats)
        
        self.feature_names.extend([f'social_feat_{i}' for i in range(10)])
        
        return features
    
    def _extract_statistical_features(self, data):
        """Extract statistical features from the sequence"""
        features = []
        
        # For each keypoint coordinate, extract statistical features
        keypoint_cols = [col for col in data.columns if any(part in col for part in ['x', 'y'])]
        
        for col in keypoint_cols[:4]:  # Use first 4 keypoints to avoid too many features
            values = data[col].values
            features.extend([
                np.mean(values), np.std(values), np.min(values), np.max(values),
                stats.skew(values), stats.kurtosis(values)
            ])
        
        self.feature_names.extend([f'{col}_{stat}' for col in keypoint_cols[:4] 
                                 for stat in ['mean', 'std', 'min', 'max', 'skew', 'kurtosis']])
        
        return features
    
    def _calculate_velocity(self, x_coords, y_coords):
        """Calculate velocity between consecutive frames"""
        if len(x_coords) < 2:
            return [0]
        
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        velocities = np.sqrt(dx**2 + dy**2)
        
        return velocities
    
    def get_feature_names(self):
        return self.feature_names