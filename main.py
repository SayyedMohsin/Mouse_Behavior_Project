import os
import sys
import argparse

# Add src to path
sys.path.append('src')

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'models', 
        'submissions',
        'notebooks',
        'src'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def train_model():
    """Train the model"""
    try:
        from train import main as train_main
        train_main()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("🔄 Please run: python src/train.py")

def predict_model():
    """Make predictions and create submission"""
    try:
        from inference import create_submission
        create_submission()
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print("🔄 Please run: python src/inference.py")

def main():
    parser = argparse.ArgumentParser(description='Mouse Behavior Classification')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['setup', 'train', 'predict', 'all'],
                       help='Mode to run: setup, train, predict, or all')
    
    args = parser.parse_args()
    
    print("🐭 Mouse Behavior Classification Pipeline")
    print("=" * 50)
    
    if args.mode == 'setup':
        print("📁 Setting up project directories...")
        setup_directories()
        print("✅ Setup completed successfully!")
        
    elif args.mode == 'train':
        print("🎯 Starting training...")
        train_model()
        
    elif args.mode == 'predict':
        print("🔮 Starting prediction...")
        predict_model()
        
    elif args.mode == 'all':
        print("🚀 Running complete pipeline...")
        setup_directories()
        train_model()
        predict_model()
        print("✅ Complete pipeline finished!")

if __name__ == '__main__':
    main()