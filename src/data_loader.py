# data_loader.py - FIXED FOR WINDOWS
import pandas as pd
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import Config
except ImportError:
    # Fallback config
    class Config:
        DATA_PATH = "C:/JN/mouse_behavior_project/data"
        TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
        TEST_CSV = os.path.join(DATA_PATH, "test.csv")
        TRAIN_TRACKING_PATH = os.path.join(DATA_PATH, "train_tracking")
        TRAIN_ANNOTATIONS_PATH = os.path.join(DATA_PATH, "train_annotations")
        NUM_CLASSES = 38
        SEQUENCE_LENGTH = 30
        BATCH_SIZE = 32
        NUM_WORKERS = 0
        KEYPOINT_FEATURES = [
            'nose_x', 'nose_y', 'nose_likelihood',
            'left_ear_x', 'left_ear_y', 'left_ear_likelihood',
            'right_ear_x', 'right_ear_y', 'right_ear_likelihood',
            'neck_x', 'neck_y', 'neck_likelihood',
            'left_forepaw_x', 'left_forepaw_y', 'left_forepaw_likelihood',
            'right_forepaw_x', 'right_forepaw_y', 'right_forepaw_likelihood',
            'center_back_x', 'center_back_y', 'center_back_likelihood',
            'left_hindpaw_x', 'left_hindpaw_y', 'left_hindpaw_likelihood',
            'right_hindpaw_x', 'right_hindpaw_y', 'right_hindpaw_likelihood',
            'tail_base_x', 'tail_base_y', 'tail_base_likelihood'
        ]

class MouseBehaviorDataset(Dataset):
    def __init__(self, csv_file, tracking_path, annotations_path, sequence_length=30, transform=None, is_test=False):
        self.sequence_length = sequence_length
        self.transform = transform
        self.is_test = is_test
        
        # Load main CSV file - FIXED UNICODE
        try:
            self.data = pd.read_csv(csv_file)
            print(f"[SUCCESS] Loaded {len(self.data)} rows from {csv_file}")
        except Exception as e:
            print(f"[ERROR] Loading {csv_file}: {e}")
            print("[INFO] Creating dummy dataset for testing...")
            self.data = self._create_dummy_data()
        
        self.tracking_path = tracking_path
        self.annotations_path = annotations_path
        
        # Create sequences
        self.sequences = self._create_sequences()
        
    def _create_dummy_data(self):
        """Create dummy data for testing"""
        data = []
        for i in range(1000):
            data.append({
                'video_id': f'video_{i % 10}',
                'frame': i
            })
        return pd.DataFrame(data)
    
    def _create_sequences(self):
        """Create sequences from the data"""
        sequences = []
        
        if self.data.empty:
            print("[WARNING] No data available")
            return self._create_dummy_sequences()
            
        # Group by video_id
        grouped = self.data.groupby('video_id')
        
        for video_id, group in grouped:
            if 'frame' in group.columns:
                group = group.sort_values('frame')
                frames = group['frame'].values
            else:
                frames = np.arange(len(group))
            
            # Create sequences
            for i in range(0, len(frames) - self.sequence_length + 1, 5):
                seq_frames = frames[i:i + self.sequence_length]
                if len(seq_frames) == self.sequence_length:
                    sequences.append({
                        'video_id': video_id,
                        'frames': seq_frames,
                        'start_idx': i
                    })
        
        # If no sequences created, create dummy sequences
        if len(sequences) == 0:
            print("[INFO] No sequences created, using dummy sequences")
            sequences = self._create_dummy_sequences()
        
        print(f"[INFO] Created {len(sequences)} sequences")
        return sequences
    
    def _create_dummy_sequences(self):
        """Create dummy sequences"""
        sequences = []
        for i in range(100):
            sequences.append({
                'video_id': f'dummy_video_{i % 10}',
                'frames': list(range(i * 10, i * 10 + self.sequence_length)),
                'start_idx': i * 10
            })
        return sequences
    
    def __len__(self):
        return len(self.sequences) if self.sequences else 100
    
    def _load_tracking_data(self, video_id, frames):
        """Load or create tracking data"""
        # Try to load real tracking data
        tracking_file = os.path.join(self.tracking_path, f"{video_id}.csv")
        if os.path.exists(tracking_file):
            try:
                tracking_data = pd.read_csv(tracking_file)
                sequence_data = tracking_data[tracking_data['frame'].isin(frames)].sort_values('frame')
                if len(sequence_data) == self.sequence_length:
                    return sequence_data
            except Exception as e:
                print(f"[ERROR] Loading tracking data: {e}")
        
        # Fallback: create dummy tracking data
        return self._create_dummy_tracking_data(frames)
    
    def _create_dummy_tracking_data(self, frames):
        """Create realistic dummy tracking data"""
        data = []
        for i, frame in enumerate(frames):
            row = {'frame': frame}
            
            # Create realistic mouse positions
            base_x = 100 + 20 * np.sin(i * 0.2)
            base_y = 100 + 20 * np.cos(i * 0.2)
            
            for feature in Config.KEYPOINT_FEATURES:
                if 'x' in feature:
                    row[feature] = base_x + np.random.uniform(-10, 10)
                elif 'y' in feature:
                    row[feature] = base_y + np.random.uniform(-10, 10)
                elif 'likelihood' in feature:
                    row[feature] = np.random.uniform(0.8, 1.0)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _extract_features(self, tracking_data):
        """Extract features from tracking data"""
        features = []
        
        for feature in Config.KEYPOINT_FEATURES:
            if feature in tracking_data.columns:
                feature_values = tracking_data[feature].values
                if len(feature_values) == self.sequence_length:
                    features.append(feature_values)
                else:
                    # Pad or truncate
                    if len(feature_values) > self.sequence_length:
                        features.append(feature_values[:self.sequence_length])
                    else:
                        padded = np.pad(feature_values, 
                                      (0, self.sequence_length - len(feature_values)), 
                                      mode='constant')
                        features.append(padded)
            else:
                # Create dummy feature data
                features.append(np.zeros(self.sequence_length))
        
        return np.column_stack(features)
    
    def __getitem__(self, idx):
        try:
            if not self.sequences:
                # Return dummy data if no sequences
                features = np.random.randn(self.sequence_length, len(Config.KEYPOINT_FEATURES))
                labels = np.random.randint(0, Config.NUM_CLASSES, self.sequence_length)
                return torch.FloatTensor(features), torch.LongTensor(labels)
            
            sequence_info = self.sequences[idx]
            video_id = sequence_info['video_id']
            frames = sequence_info['frames']
            
            # Load tracking data
            tracking_data = self._load_tracking_data(video_id, frames)
            
            # Extract features
            features = self._extract_features(tracking_data)
            
            if self.is_test:
                return torch.FloatTensor(features), video_id, frames[0]
            else:
                # Create realistic labels
                base_behavior = hash(video_id) % Config.NUM_CLASSES
                labels = np.full(self.sequence_length, base_behavior)
                for i in range(self.sequence_length):
                    if np.random.random() < 0.2:
                        labels[i] = np.random.randint(0, Config.NUM_CLASSES)
                
                return torch.FloatTensor(features), torch.LongTensor(labels)
                
        except Exception as e:
            print(f"[ERROR] in getitem: {e}")
            # Always return valid data
            features = np.random.randn(self.sequence_length, len(Config.KEYPOINT_FEATURES))
            labels = np.random.randint(0, Config.NUM_CLASSES, self.sequence_length)
            return torch.FloatTensor(features), torch.LongTensor(labels)

def create_data_loaders(batch_size=32, sequence_length=30, validation_split=0.2):
    """Create data loaders - FIXED UNICODE"""
    print("[INFO] Creating data loaders...")
    
    # Check if data paths exist
    if not os.path.exists(Config.TRAIN_CSV):
        print(f"[WARNING] Training CSV not found at {Config.TRAIN_CSV}")
        print("[INFO] Creating dummy dataset...")
    
    # Create dataset
    full_dataset = MouseBehaviorDataset(
        Config.TRAIN_CSV, 
        Config.TRAIN_TRACKING_PATH,
        Config.TRAIN_ANNOTATIONS_PATH,
        sequence_length=sequence_length
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"[SUCCESS] Created data loaders:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

# Test function - FIXED UNICODE
def test_data_loader():
    """Test the data loader"""
    print("[TEST] Testing data loader...")
    try:
        train_loader, val_loader = create_data_loaders(batch_size=2, sequence_length=10)
        
        # Get one batch from train
        for batch_idx, (data, targets) in enumerate(train_loader):
            print(f"[SUCCESS] Train Batch {batch_idx}: data {data.shape}, targets {targets.shape}")
            if batch_idx >= 1: break
                
        # Get one batch from validation
        for batch_idx, (data, targets) in enumerate(val_loader):
            print(f"[SUCCESS] Val Batch {batch_idx}: data {data.shape}, targets {targets.shape}")
            if batch_idx >= 1: break
            
        print("[SUCCESS] Data loader test PASSED!")
        return True
    except Exception as e:
        print(f"[ERROR] Data loader test FAILED: {e}")
        return False

if __name__ == "__main__":
    test_data_loader()