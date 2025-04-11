import os
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class CorruptionDataset(Dataset):
    def __init__(self, clean_root, corrupted_root, transform=None):
        self.transform = transform

        # Match multiple image extensions
        clean_paths = glob(os.path.join(clean_root, '**', '*.*'), recursive=True)
        clean_paths = [p for p in clean_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
        clean_labels = [0] * len(clean_paths)

        corrupted_paths = glob(os.path.join(corrupted_root, '*.*'))
        corrupted_paths = [p for p in corrupted_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
        corrupted_labels = [1] * len(corrupted_paths)

        self.image_paths = clean_paths + corrupted_paths
        self.labels = clean_labels + corrupted_labels

        print(f"Loaded {len(clean_paths)} clean and {len(corrupted_paths)} corrupted images.")
        if len(self.image_paths) == 0:
            print("No images found! Double-check folder paths and image formats.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# Transform (resize + normalize for pretrained ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def train_classifier():
    # Paths
    clean_root = './data/classifier/train/clean/'
    corrupted_root = './data/classifier/train/corrupted/'

    # Dataset and loaders
    full_dataset = CorruptionDataset(clean_root, corrupted_root, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    print("Total dataset size:", len(full_dataset))
    print("Example image path:", full_dataset.image_paths[0] if len(full_dataset) > 0 else "No images found!")
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Binary output
    model = model.to(device)

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop with tqdm
    epochs=2
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch+1}/{epochs}")
        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs).squeeze() > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), 'corruption_classifier.pth')
    print("Model saved as corruption_classifier.pth")

def test_classifier():
    # Paths
    clean_root = './data/classifier/test/clean/images1024x1024/'
    corrupted_root = './data/classifier/test/corrupted/'

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # Dataset and loader
    test_dataset = CorruptionDataset(clean_root, corrupted_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load("corruption_classifier.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # Inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", ncols=100):
            images = images.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs).squeeze() > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Clean", "Corrupted"]))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Pred: Clean", "Pred: Corrupted"], yticklabels=["True: Clean", "True: Corrupted"])
    plt.title("Confusion Matrix")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.show()

train_classifier()