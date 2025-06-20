import matplotlib.pyplot as plt
import csv
import sys

if len(sys.argv) < 2:
    print("Usage: python plot_metrics <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

epochs = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    i = 0
    for row in reader:
        epochs.append(int(row['epoch']))
        train_loss.append(float(row['train_loss']))
        val_loss.append(float(row['val_loss']))
        train_acc.append(float(row['train_acc']))
        val_acc.append(float(row['val_acc']))


plt.figure(figsize=(10, 6))

# Plot loss
plt.plot(epochs, train_loss, color='blue', label='Train Loss')
plt.plot(epochs, val_loss, color='red', label='Val Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_plot.png')
plt.show()


plt.figure(figsize=(10, 6))


# Plot accuracy
plt.plot(epochs, train_acc, color='blue', label='Train Acc')
plt.plot(epochs, val_acc, color='red', label='Val Acc')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_plot.png')
plt.show()
