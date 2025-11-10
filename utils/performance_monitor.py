import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

class PerformanceMonitor:
    def __init__(self):
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot detailed confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def generate_detailed_report(self, y_true, y_pred, class_names):
        """Generate detailed performance report"""
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names, output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('detailed_classification_report.csv')
        
        # Plot per-class performance
        class_metrics = report_df.iloc[:-3, :]  # Exclude averages
        
        plt.figure(figsize=(12, 6))
        class_metrics[['precision', 'recall', 'f1-score']].plot(kind='bar')
        plt.title('Per-Class Performance Metrics')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('per_class_performance.png')
        plt.close()
        
        return report_df