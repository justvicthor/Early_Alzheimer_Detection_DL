import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ADNIDataset
from model import ClassifierCNN
import os
from tqdm import tqdm

# TODO use argparse for dynamic set

SCANS_DIR = "../ADNI_processed"  # Path to the directory containing the scans
#TRAIN_TSV = "../vitto/Train_50.tsv"
TRAIN_TSV = "./participants_Train50_updated.tsv" # new path for training data 
#VAL_TSV = "../vitto/Val_50.tsv"
VAL_TSV = "./participants_Val50_updated.tsv" # new path for validation data

# Training params
NUM_CLASSES = 3
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

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
    
    return running_loss / len(val_loader), 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = ADNIDataset(SCANS_DIR, TRAIN_TSV, mode='train', num_classes=NUM_CLASSES)
    val_dataset = ADNIDataset(SCANS_DIR, VAL_TSV, mode='val', num_classes=NUM_CLASSES)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    

    model = ClassifierCNN(num_classes=NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # training!
    best_val_acc = 0
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved new best model!')

    print('Training complete. Best validation accuracy:', best_val_acc)

if __name__ == '__main__':
    main() 