import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
csv_path = "C:/Users/natha/OneDrive/Documents/ECE1512/ProjectB/ECE1512 Project B/ECE1512 Project B/results/bracs_medical_ssl_config_log.csv"


df = pd.read_csv(csv_path)

# Epochs
epochs = df['epoch']

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, df['val_loss'], label='Validation Loss', marker='o')
plt.plot(epochs, df['test_loss'], label='Test Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('loss_curves.png', dpi=300)
plt.show()

# Plot AUROC
plt.figure(figsize=(10, 5))
plt.plot(epochs, df['val_auc'], label='Validation AUROC', marker='o')
plt.plot(epochs, df['test_auc'], label='Test AUROC', marker='x')
plt.xlabel('Epoch')
plt.ylabel('AUROC')
plt.title('AUROC Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('auroc_curves.png', dpi=300)
plt.show()

# Optional: Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, df['val_acc'], label='Validation Accuracy', marker='o')
plt.plot(epochs, df['test_acc'], label='Test Accuracy', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('accuracy_curves.png', dpi=300)
plt.show()
