import numpy as np
import torch
from scipy import signal

class AdvancedTemporalSmoothing:
    def __init__(self):
        self.smoothing_methods = {}
    
    def median_filter(self, predictions, window_size=5):
        """Apply median filtering to temporal predictions"""
        if len(predictions) < window_size:
            return predictions
        
        smoothed = []
        for i in range(len(predictions)):
            start = max(0, i - window_size // 2)
            end = min(len(predictions), i + window_size // 2 + 1)
            window = predictions[start:end]
            
            # Get most common prediction in window
            most_common = np.argmax(np.bincount(window))
            smoothed.append(most_common)
        
        return np.array(smoothed)
    
    def hidden_markov_smoothing(self, predictions, probabilities, transition_matrix=None):
        """Apply HMM-based temporal smoothing"""
        if transition_matrix is None:
            # Create default transition matrix (favor staying in same state)
            n_classes = probabilities.shape[1]
            transition_matrix = np.eye(n_classes) * 0.7 + np.ones((n_classes, n_classes)) * 0.3 / n_classes
        
        # Viterbi algorithm for HMM smoothing
        n_states = probabilities.shape[1]
        n_observations = len(predictions)
        
        # Initialize Viterbi table
        viterbi = np.zeros((n_states, n_observations))
        backpointers = np.zeros((n_states, n_observations), dtype=int)
        
        # Initialization
        viterbi[:, 0] = np.log(probabilities[0] + 1e-8)
        
        # Recursion
        for t in range(1, n_observations):
            for s in range(n_states):
                transition_probs = viterbi[:, t-1] + np.log(transition_matrix[:, s] + 1e-8)
                best_prev_state = np.argmax(transition_probs)
                viterbi[s, t] = transition_probs[best_prev_state] + np.log(probabilities[t, s] + 1e-8)
                backpointers[s, t] = best_prev_state
        
        # Backtracking
        smoothed = np.zeros(n_observations, dtype=int)
        smoothed[-1] = np.argmax(viterbi[:, -1])
        
        for t in range(n_observations-2, -1, -1):
            smoothed[t] = backpointers[smoothed[t+1], t+1]
        
        return smoothed
    
    def confidence_based_smoothing(self, predictions, confidence_scores, threshold=0.8):
        """Smooth predictions based on confidence scores"""
        smoothed = predictions.copy()
        
        for i in range(1, len(predictions)-1):
            if confidence_scores[i] < threshold:
                # Low confidence - look at neighbors
                neighbors = predictions[i-1:i+2]
                if predictions[i-1] == predictions[i+1] and confidence_scores[i-1] > threshold:
                    smoothed[i] = predictions[i-1]
        
        return smoothed
    
    def behavioral_constraint_smoothing(self, predictions, behavior_constraints):
        """Apply behavior-specific temporal constraints"""
        smoothed = predictions.copy()
        
        for i in range(1, len(predictions)):
            current_behavior = predictions[i]
            prev_behavior = predictions[i-1]
            
            # Check if transition is allowed
            if not self._is_transition_allowed(prev_behavior, current_behavior, behavior_constraints):
                # Find most likely allowed transition
                smoothed[i] = self._find_best_allowed_transition(prev_behavior, behavior_constraints)
        
        return smoothed
    
    def _is_transition_allowed(self, prev_behavior, current_behavior, constraints):
        """Check if behavior transition is allowed"""
        # Implement behavior transition constraints
        # Example: Cannot directly go from "sleeping" to "running"
        forbidden_transitions = {
            'sleeping': ['running', 'fighting'],
            'eating': ['running', 'jumping']
            # Add more based on domain knowledge
        }
        
        return True  # Placeholder
    
    def _find_best_allowed_transition(self, prev_behavior, constraints):
        """Find the best allowed behavior transition"""
        # Implement logic to find most probable allowed transition
        return prev_behavior  # Placeholder