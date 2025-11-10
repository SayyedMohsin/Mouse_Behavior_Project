import torch
import argparse
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'tune', 'ensemble', 'full_pipeline'])
    args = parser.parse_args()
    
    print("🚀 Advanced Training Pipeline")
    print("=" * 50)
    
    # Set device - FIX CUDA ISSUE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    if args.mode == 'tune':
        print("🎯 Starting hyperparameter tuning...")
        try:
            from data_loader import create_data_loaders
            train_loader, val_loader = create_data_loaders(batch_size=16)
            
            # Import after data loader to avoid circular imports
            from advanced_features.hyperparameter_tuning import HyperparameterOptimizer
            optimizer = HyperparameterOptimizer(train_loader, val_loader, device)
            best_params = optimizer.optimize(n_trials=3)  # Reduced for testing
            print(f"✅ Best parameters: {best_params}")
        except Exception as e:
            print(f"❌ Hyperparameter tuning failed: {e}")
            print("🔄 Using default parameters...")
    
    elif args.mode == 'ensemble':
        print("🤖 Creating advanced ensemble...")
        try:
            from ensemble.model_ensemble import AdvancedEnsemble
            
            # Check which models exist
            model_configs = []
            possible_models = [
                'models/best_model.pth',
                'models/checkpoint_epoch_1.pth',
                'models/checkpoint_epoch_5.pth'
            ]
            
            for model_path in possible_models:
                if os.path.exists(model_path):
                    model_configs.append({
                        'type': 'base', 
                        'path': model_path, 
                        'weight': 1.0
                    })
            
            if model_configs:
                ensemble = AdvancedEnsemble(model_configs, device)
                print(f"✅ Ensemble created with {len(model_configs)} models!")
            else:
                print("⚠ No trained models found for ensemble")
                
        except Exception as e:
            print(f"❌ Ensemble creation failed: {e}")
    
    elif args.mode == 'full_pipeline':
        print("🔄 Running full advanced pipeline...")
        
        # 1. Hyperparameter tuning
        print("Step 1: Hyperparameter tuning...")
        try:
            from data_loader import create_data_loaders
            from advanced_features.hyperparameter_tuning import HyperparameterOptimizer
            
            train_loader, val_loader = create_data_loaders(batch_size=16)
            optimizer = HyperparameterOptimizer(train_loader, val_loader, device)
            best_params = optimizer.optimize(n_trials=2)
            print(f"✓ Best parameters: {best_params}")
        except Exception as e:
            print(f"⚠ Tuning skipped: {e}")
        
        # 2. Train with best parameters
        print("Step 2: Training...")
        try:
            from train import main as train_main
            train_main()
        except Exception as e:
            print(f"⚠ Training skipped: {e}")
        
        # 3. Create ensemble
        print("Step 3: Creating ensemble...")
        try:
            from ensemble.model_ensemble import AdvancedEnsemble
            model_configs = [{'type': 'base', 'path': 'models/best_model.pth', 'weight': 1.0}]
            ensemble = AdvancedEnsemble(model_configs, device)
            print("✓ Ensemble created")
        except Exception as e:
            print(f"⚠ Ensemble skipped: {e}")
        
        # 4. Create submission
        print("Step 4: Creating submission...")
        try:
            from inference import create_submission
            create_submission()
        except Exception as e:
            print(f"⚠ Submission skipped: {e}")
        
        print("✅ Full pipeline completed!")

if __name__ == '__main__':
    main()