import argparse, yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ADNIDataset
from model import ClassifierCNN
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import random

# Set random seeds for reproducibility
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
cfg_path = parser.parse_args().config        
with open(cfg_path, 'r') as f:               
    cfg = yaml.safe_load(f)

SCANS_DIR    = cfg['data']['scans_dir']
TEST_TSV     = cfg['data']['test_tsv']
MODEL_PATH   = cfg['file_name'] + '.pth'       
NUM_CLASSES  = cfg['model']['num_classes']
BATCH_SIZE = cfg['data']['test_batch_size']   
CROP_SIZE = cfg['data']['crop_size']
NUM_WORKERS = cfg['data']['workers']

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = (correct / total) * 100
    auc = roc_auc_score(
        all_labels,
        all_probs,
        multi_class='ovr',
        average='macro'
    )
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    per_class_auc = roc_auc_score(
    all_labels,
    all_probs,
    multi_class='ovr',
    average=None 
    )

    print(f"Test Accuracy (simple): {acc:.2f}%")
    print("Per-class AUC:", per_class_auc)
    print(f"Balanced Accuracy     : {bal_acc*100:.2f}%")
    print(f"Macro-AUC             : {auc:.4f}")
    return all_preds, all_labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = ADNIDataset(
        SCANS_DIR, TEST_TSV,
        mode='test',
        num_classes=NUM_CLASSES,
        use_augmentation=False,
        crop_size=CROP_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = ClassifierCNN(
           in_channels = cfg['model']['in_channels'],
           num_classes = NUM_CLASSES,
           expansion   = cfg['model']['expansion'],
           feature_dim = cfg['model']['feature_dim'],
           nhid        = cfg['model']['nhid'],
           norm_type   = cfg['model']['norm_type'],
           crop_size   = CROP_SIZE
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    preds, labels = test(model, test_loader, device)

    # Save predictions in the same format as the input participants file (tab-separated, same columns, plus true_label and predicted_label at the end)
    filename = "test_predictions.tsv"
    df = pd.read_csv(TEST_TSV, sep='\t')
    map_classes = {0: "CN", 1: "LMCI", 2: "AD"} if NUM_CLASSES == 3 else {0: "CN", 1: "AD"}
    preds = [map_classes[p] for p in preds]
    labels = [map_classes[l] for l in labels]

    df['true_label'] = labels
    df['predicted_label'] = preds
    # Ensure columns order: all original columns, then true_label, predicted_label
    cols = list(df.columns)
    # Move true_label and predicted_label to the end if not already
    for col in ['true_label', 'predicted_label']:
        if col in cols:
            cols.remove(col)
            cols.append(col)
    df = df[cols]
    df.to_csv(filename, sep='\t', index=False)
    print(f"Predictions saved to {filename}")

if __name__ == "__main__":
    main()