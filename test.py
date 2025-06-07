import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ADNIDataset
from model import ClassifierCNN
import pandas as pd
from tqdm import tqdm

SCANS_DIR = "../ADNI_processed"
TEST_TSV = "./participants_Test50.tsv" 
MODEL_PATH = "best_model.pth"

# Params
NUM_CLASSES = 3
BATCH_SIZE = 8

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

    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Test AUC: {auc:.4f}")
    return all_preds, all_labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = ADNIDataset(SCANS_DIR, TEST_TSV, mode='val', num_classes=NUM_CLASSES)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ClassifierCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    preds, labels = test(model, test_loader, device)

    # Save predictions
    filename = "test_predictions.tsv"
    df = pd.read_csv(TEST_TSV, sep='\t')
    df['predicted_label'] = preds
    df['true_label'] = labels
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

if __name__ == "__main__":
    main()