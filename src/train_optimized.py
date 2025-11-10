import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import Config
except ImportError:
    class Config:
        NUM_CLASSES = 38
        SEQUENCE_LENGTH = 30
        BATCH_SIZE = 32
        # OPTIMIZED PARAMETERS FROM TUNING
        LEARNING_RATE = 0.00583  # From tuning
        HIDDEN_SIZE = 128       # From tuning  
        DROPOUT_RATE = 0.146    # From tuning
        EPOCHS = 100
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OPTIMIZED MODEL with tuned parameters
class OptimizedMouseModel(nn.Module):
    def __init__(self, num_classes=38, input_dim=30, hidden_dim=128, dropout_rate=0.146):
        super(OptimizedMouseModel, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # Optimized architecture with tuned parameters
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Process each time step
        features_processed = []
        for t in range(seq_len):
            frame_features = x[:, t, :]
            encoded = self.encoder(frame_features)
            features_processed.append(encoded)
        
        # Average over time
        combined = torch.stack(features_processed, dim=1).mean(dim=1)
        output = self.classifier(combined)
        
        return output

class OptimizedTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets[:, 0])  # Use first frame target
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets[:, 0]).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets[:, 0])
                running_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets[:, 0]).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, epochs):
        print(f"🚀 Starting OPTIMIZED training on {self.device}...")
        print(f"🎯 Optimized Parameters:")
        print(f"   - Learning Rate: {Config.LEARNING_RATE}")
        print(f"   - Hidden Size: {Config.HIDDEN_SIZE}")
        print(f"   - Dropout Rate: {Config.DROPOUT_RATE}")
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 60)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f'✅ Epoch {epoch+1}/{epochs}:')
            print(f'   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'models/optimized_model.pth')
                print(f'💾 BEST MODEL SAVED! Val Loss: {val_loss:.4f}')
            
            # Early stopping check
            if epoch > 10 and val_loss > min(self.val_losses[-5:]):
                print("🛑 Early stopping triggered!")
                break
        
        print("🎯 Optimized training completed!")
        self.plot_training_progress()
    
    def plot_training_progress(self):
        """Plot training progress"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, label='Train Loss')
            plt.plot(self.val_losses, label='Val Loss')
            plt.title('Training & Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(self.val_accuracies)
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            
            plt.tight_layout()
            plt.savefig('models/optimized_training.png')
            plt.close()
            
            print("📊 Training progress plot saved!")
        except:
            print("📊 Matplotlib not available, skipping plot")

def main():
    """Main optimized training function"""
    print("🎯 OPTIMIZED Mouse Behavior Classification Training")
    print("=" * 60)
    
    device = torch.device(Config.DEVICE)
    print(f"🖥️ Using device: {device}")
    
    try:
        from data_loader import create_data_loaders
        
        # Create data loaders
        print("📁 Loading data...")
        train_loader, val_loader = create_data_loaders(
            batch_size=Config.BATCH_SIZE,
            sequence_length=Config.SEQUENCE_LENGTH
        )
        
        # Create OPTIMIZED model
        print("🧠 Creating optimized model...")
        input_dim = 30  # Number of keypoint features
        model = OptimizedMouseModel(
            num_classes=Config.NUM_CLASSES,
            input_dim=input_dim,
            hidden_dim=Config.HIDDEN_SIZE,
            dropout_rate=Config.DROPOUT_RATE
        )
        
        print(f"📐 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer and start training
        trainer = OptimizedTrainer(model, train_loader, val_loader, device)
        trainer.train(Config.EPOCHS)
        
        print("🎉 Optimized training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in optimized training: {e}")
        fallback_training()

def fallback_training():
    """Fallback training"""
    print("🛡️ Starting fallback training...")
    
    # Simple training as backup
    from train import main as simple_main
    simple_main()

if __name__ == '__main__':
    main()