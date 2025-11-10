import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class AdvancedSocialFeatures:
    def __init__(self):
        self.feature_names = []
    
    def extract_multi_mouse_features(self, mice_data):
        """Extract advanced social interaction features between multiple mice"""
        features = []
        
        if len(mice_data) < 2:
            # Return zeros if only one mouse
            return np.zeros(20)
        
        # 1. Distance-based features
        distance_features = self._extract_distance_features(mice_data)
        features.extend(distance_features)
        
        # 2. Movement correlation features
        movement_features = self._extract_movement_correlation(mice_data)
        features.extend(movement_features)
        
        # 3. Social hierarchy features
        hierarchy_features = self._extract_social_hierarchy(mice_data)
        features.extend(hierarchy_features)
        
        # 4. Interaction zone features
        zone_features = self._extract_interaction_zones(mice_data)
        features.extend(zone_features)
        
        return np.array(features)
    
    def _extract_distance_features(self, mice_data):
        """Extract distance-based social features"""
        features = []
        
        # Calculate pairwise distances
        positions = []
        for mouse_data in mice_data:
            com_x = mouse_data['center_back_x'].mean()
            com_y = mouse_data['center_back_y'].mean()
            positions.append([com_x, com_y])
        
        positions = np.array(positions)
        distances = cdist(positions, positions)
        
        # Minimum distance between any two mice
        min_distance = np.min(distances[distances > 0])
        features.append(min_distance)
        
        # Average distance
        avg_distance = np.sum(distances) / (len(distances) * (len(distances)-1))
        features.append(avg_distance)
        
        # Distance variance
        features.append(np.var(distances[distances > 0]))
        
        # Time spent in close proximity (< 50 pixels)
        close_proximity_time = np.mean(distances < 50)
        features.append(close_proximity_time)
        
        self.feature_names.extend([
            'min_inter_mouse_distance', 'avg_inter_mouse_distance',
            'distance_variance', 'close_proximity_ratio'
        ])
        
        return features
    
    def _extract_movement_correlation(self, mice_data):
        """Extract movement correlation features"""
        features = []
        
        velocities = []
        for mouse_data in mice_data:
            vx = np.diff(mouse_data['center_back_x'])
            vy = np.diff(mouse_data['center_back_y'])
            vel = np.sqrt(vx**2 + vy**2)
            velocities.append(vel)
        
        if len(velocities) >= 2:
            # Velocity correlation
            corr_matrix = np.corrcoef(velocities)
            features.extend([
                np.mean(corr_matrix),  # Average correlation
                np.max(corr_matrix),   # Maximum correlation
                np.std(corr_matrix)    # Correlation variability
            ])
        else:
            features.extend([0, 0, 0])
        
        self.feature_names.extend([
            'avg_velocity_correlation', 'max_velocity_correlation',
            'velocity_correlation_std'
        ])
        
        return features
    
    def _extract_social_hierarchy(self, mice_data):
        """Extract social hierarchy features"""
        features = []
        
        # Calculate activity levels
        activity_levels = []
        for mouse_data in mice_data:
            movement = np.sqrt(
                np.diff(mouse_data['center_back_x'])**2 + 
                np.diff(mouse_data['center_back_y'])**2
            )
            activity_levels.append(np.mean(movement))
        
        if activity_levels:
            # Dominance indicators
            features.extend([
                np.max(activity_levels) - np.min(activity_levels),  # Activity difference
                np.std(activity_levels),  # Activity inequality
                np.argmax(activity_levels)  # Most active mouse index
            ])
        else:
            features.extend([0, 0, 0])
        
        self.feature_names.extend([
            'activity_difference', 'activity_inequality', 'dominant_mouse_idx'
        ])
        
        return features
    
    def _extract_interaction_zones(self, mice_data):
        """Extract interaction zone features"""
        features = []
        
        # Define interaction zones (personal space)
        zone_ratios = []
        for mouse_data in mice_data:
            mouse_x = mouse_data['center_back_x'].mean()
            mouse_y = mouse_data['center_back_y'].mean()
            
            # Check if other mice are in personal zone (30 pixels)
            personal_zone_violations = 0
            for other_mouse in mice_data:
                if other_mouse is mouse_data:
                    continue
                
                other_x = other_mouse['center_back_x'].mean()
                other_y = other_mouse['center_back_y'].mean()
                distance = np.sqrt((mouse_x - other_x)**2 + (mouse_y - other_y)**2)
                
                if distance < 30:
                    personal_zone_violations += 1
            
            zone_ratios.append(personal_zone_violations)
        
        features.extend([
            np.mean(zone_ratios),
            np.max(zone_ratios),
            np.sum(zone_ratios) > 0  # Any interactions
        ])
        
        self.feature_names.extend([
            'avg_zone_violations', 'max_zone_violations', 'any_interaction'
        ])
        
        return features