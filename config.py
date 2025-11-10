import os
import torch

class Config:
    # Paths
    DATA_PATH = "C:/JN/mouse_behavior_project/data"
    TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
    TEST_CSV = os.path.join(DATA_PATH, "test.csv")
    TRAIN_TRACKING_PATH = os.path.join(DATA_PATH, "train_tracking")
    TRAIN_ANNOTATIONS_PATH = os.path.join(DATA_PATH, "train_annotations")
    
    # Model parameters
    NUM_CLASSES = 38
    SEQUENCE_LENGTH = 30
    FEATURE_DIM = 128
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    NUM_WORKERS = 4
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Features to extract
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

config = Config()