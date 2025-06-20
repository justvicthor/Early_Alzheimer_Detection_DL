import argparse, yaml
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
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml')
cfg_path = parser.parse_args().config        # <-- nome file
with open(cfg_path, 'r') as f:               # <-- apertura UNA sola volta
    cfg = yaml.safe_load(f)

SCANS_DIR = cfg['data']['scans_dir']
TRAIN_TSV = cfg['data']['train_tsv']
VAL_TSV   = cfg['data']['val_tsv']

# Training params
NUM_CLASSES   = cfg['model']['num_classes']
BATCH_SIZE    = cfg['data']['batch_size']
VAL_BATCH_SZ  = cfg['data']['val_batch_size']
NUM_EPOCHS    = cfg['training']['epochs']
LEARNING_RATE = cfg['training']['optimizer']['lr']
CROP_SIZE = cfg['data']['crop_size']
BLUR_SIGMA = cfg['data']['blur_sigma']
NUM_WORKERS = cfg['data']['workers']


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
        if cfg['training']['gradient_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        cfg['training']['gradient_clip'])
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
    
    train_dataset = ADNIDataset(
            SCANS_DIR, TRAIN_TSV,
            mode='train',
            num_classes=NUM_CLASSES,
            use_augmentation=cfg['data']['use_augmentation'],
            crop_size=CROP_SIZE,
            blur_sigma_range=tuple(BLUR_SIGMA)
    )
    val_dataset = ADNIDataset(
            SCANS_DIR, VAL_TSV,
            mode='val',
            num_classes=NUM_CLASSES,
            use_augmentation=False,
            crop_size=CROP_SIZE)
    
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SZ, shuffle=False, num_workers=NUM_WORKERS)
    

    model = ClassifierCNN(
           in_channels = cfg['model']['in_channels'],
           num_classes = cfg['model']['num_classes'],
           expansion   = cfg['model']['expansion'],
           feature_dim = cfg['model']['feature_dim'],
           nhid        = cfg['model']['nhid'],
           norm_type   = cfg['model']['norm_type'],
           crop_size   = cfg['data']['crop_size']
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # SGD optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    best_val_loss = float("+inf")

    
    out_csv  = cfg['log_csv']                       # ./saved_model/run1_training_log.csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    csvfile  = open(out_csv, 'w', newline='')       
    writer   = csv.writer(csvfile)
    writer.writerow(['epoch', 'train_loss', 'train_acc','val_loss', 'val_acc', 'val_auc'])
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}')

        
        writer.writerow([epoch+1, train_loss, train_acc,
                 val_loss, val_acc, val_auc])
        csvfile.flush()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(cfg['file_name']), exist_ok=True)
            torch.save(model.state_dict(), cfg['file_name'] + '.pth')
            print('Saved new best model!')

    csvfile.close()
    print('Training complete. Best validation loss:', best_val_loss)


if __name__ == '__main__':
    main() 