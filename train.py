import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ADNIDataset
from model import ClassifierCNN
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
import csv


# TODO use argparse for dynamic set

# Possible improvements:
# 1. Model Architecture Adjustments
# InstanceNorm3d instead of BatchNorm3d | Smaller kernel/stride in the initial layer | Wider, fewer layers

# 2. Data Augmentation
# Random crop to 96×96×96 and Gaussian blur in your dataset class.

# 3. Solve class imbalance
# Use WeightedRandomSampler or pass class weights to CrossEntropyLoss

SCANS_DIR = "../ADNI_processed"  # Path to the directory containing the scans
#TRAIN_TSV = "../vitto/Train_50.tsv"
TRAIN_TSV = "./participants_Train50.tsv" # new path for training data 
#VAL_TSV = "../vitto/Val_50.tsv"
VAL_TSV = "./participants_Val50.tsv" # new path for validation data

# Training params
NUM_CLASSES = 3
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.001


# Solve class imbalance
def get_sample_weights(dataset):
    labels = []
    for i in range(len(dataset)):
        # dataset[i] returns (image, label)
        _, label = dataset[i]
        labels.append(label)
    labels = np.array(labels)
    class_sample_count = np.array([np.sum(labels == t) for t in np.unique(labels)])
    weight_per_class = 1. / class_sample_count
    sample_weights = weight_per_class[labels]
    return sample_weights



def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc='Training'):
        if images is None or labels is None:
            continue
            
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            if images is None or labels is None:
                continue

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())


    loss = running_loss / len(val_loader)
    acc = (correct / total) * 100
    
    auc = roc_auc_score(
        all_labels,
        all_probs,
        multi_class='ovr',
        average='macro'
    )

    return loss, acc, auc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = ADNIDataset(SCANS_DIR, TRAIN_TSV, mode='train', num_classes=NUM_CLASSES)
    val_dataset = ADNIDataset(SCANS_DIR, VAL_TSV, mode='val', num_classes=NUM_CLASSES)
    
    # # Solve class imbalance
    # sample_weights = get_sample_weights(train_dataset)
    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    

    model = ClassifierCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # SGD optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    best_val_loss = float("+inf")

    csvfile = open('training_log.csv', mode='w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc'])

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}')

        
        writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, val_auc])
        csvfile.flush()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved new best model!')

    csvfile.close()
    print('Training complete. Best validation loss:', best_val_loss)


if __name__ == '__main__':
    main() 