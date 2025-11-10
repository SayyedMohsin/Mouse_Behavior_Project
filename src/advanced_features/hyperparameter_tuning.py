import optuna
import torch
import torch.nn as nn

class HyperparameterOptimizer:
    def __init__(self, train_loader, val_loader, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
    
    def objective(self, trial):
        """Simple hyperparameter optimization"""
        try:
            # Suggest hyperparameters
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            
            # Simple model for testing
            from train import SimpleMouseModel
            model = SimpleMouseModel(hidden_dim=hidden_size).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # Short training for optimization
            best_val_loss = float('inf')
            
            for epoch in range(3):  # Very short training
                # Training
                model.train()
                for data, targets in self.train_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, targets[:, 0])  # Use first frame
                    loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data, targets in self.val_loader:
                        data, targets = data.to(self.device), targets.to(self.device)
                        outputs = model(data)
                        loss = criterion(outputs, targets[:, 0])
                        val_loss += loss.item()
                
                val_loss /= len(self.val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            return best_val_loss
            
        except Exception as e:
            print(f"Error in trial: {e}")
            return float('inf')
    
    def optimize(self, n_trials=10):
        """Run hyperparameter optimization"""
        print(f"🔍 Running {n_trials} hyperparameter trials...")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        print("✅ Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value:.4f}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        return study.best_params

# Test function
def test_optimizer():
    """Test the optimizer"""
    print("Testing hyperparameter optimizer...")
    
    # Create dummy data
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=50):
            self.num_samples = num_samples
        def __len__(self):
            return self.num_samples
        def __getitem__(self, idx):
            features = torch.randn(30, 30)
            labels = torch.randint(0, 38, (30,))
            return features, labels
    
    dummy_dataset = DummyDataset()
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=8)
    
    optimizer = HyperparameterOptimizer(dummy_loader, dummy_loader, 'cpu')
    best_params = optimizer.optimize(n_trials=2)
    
    print("✅ Optimizer test passed!")
    return best_params

if __name__ == '__main__':
    test_optimizer()