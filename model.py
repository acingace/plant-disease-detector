# Necessary imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold

# Data Pre-Processing with Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),           
    transforms.RandomHorizontalFlip(),        
    transforms.RandomRotation(20),            
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
    transforms.ToTensor(),                    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])

valid_transform = transforms.Compose([
    transforms.Resize((256, 256)),            
    transforms.ToTensor(),                    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Device Management
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_default_device()

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

print(f"Using device: {device}")

class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        self.dataset = dl.dataset  # Store reference to dataset

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

    def get_dataset_len(self):
        """Get the length of the dataset"""
        return len(self.dataset)

# Data Directory Setup
data_dir = "/Users/gracegomes/Desktop/Detector/PlantVillage"
train_dir = os.path.join(data_dir, "train")

# Load the dataset
full_dataset = ImageFolder(train_dir, transform=train_transform)

# Get the class names
diseases = full_dataset.classes

# Pretrained ResNet Model Definition
class PretrainedResNet(nn.Module):
    """A ResNet model with pretrained weights for image classification"""
    def __init__(self, num_classes):
        super().__init__()
        # Load pretrained ResNet18 model
        self.network = models.resnet18(pretrained=True)

        # Freeze all layers
        for param in self.network.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers (fine-tuning)
        for param in self.network.layer4.parameters():
            param.requires_grad = True

        # Modify the last fully connected layer to match the number of classes
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)

# Training and Validation Functions
def validate_model(model, valid_loader):
    """Validate the model"""
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device)  # Move to device
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            valid_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect all predictions and true labels for additional metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = valid_loss / len(valid_loader.dataset)
    accuracy = correct / total

    # Calculate additional metrics
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Validation - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    # print(f"Confusion Matrix:\n{conf_matrix}")  # Optional

    return {"val_loss": avg_loss, "val_accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def train_model(model, train_loader, valid_loader, epochs=10, lr=0.001):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"Epoch {epoch + 1}/{epochs} - Training started...")

        all_train_labels = []
        all_train_preds = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Collect labels and predictions for calculating metrics
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                current_loss = running_loss / ((batch_idx + 1) * train_loader.dl.batch_size)
                print(f"Epoch [{epoch + 1}], Batch [{batch_idx + 1}/{len(train_loader)}], Training Loss: {current_loss:.4f}")

        epoch_loss = running_loss / train_loader.get_dataset_len()
        train_accuracy = correct / total  
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        
        print(f"Epoch {epoch + 1}/{epochs} completed. Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}")

        valid_metrics = validate_model(model, valid_loader)
        current_f1 = valid_metrics['f1']

        if valid_metrics['val_accuracy'] < train_accuracy:
            print(f"Potential overfitting: Training Accuracy = {train_accuracy:.4f}, Validation Accuracy = {valid_metrics['val_accuracy']:.4f}")

        if current_f1 > best_f1:
            best_f1 = current_f1
            print(f"New best F1 score: {best_f1:.4f}.")

        scheduler.step()

    return best_f1

# Define your batch size and number of epochs here
batch_size = 32
num_epochs = 10
k_folds = 5

# Prepare cross-validation
kfold = KFold(n_splits=k_folds, shuffle=True)

# Lists to hold the results
fold_results = {}

print('--------------------------------')

for fold, (train_ids, valid_ids) in enumerate(kfold.split(full_dataset)):
    print(f'FOLD {fold+1}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = DeviceDataLoader(DataLoader(
                      full_dataset, 
                      batch_size=batch_size, sampler=train_subsampler), device)
    valid_loader = DeviceDataLoader(DataLoader(
                      full_dataset,
                      batch_size=batch_size, sampler=valid_subsampler), device)

    # Initialize the model
    num_classes = len(diseases)
    model = PretrainedResNet(num_classes=num_classes).to(device)

    print(f'Training for {num_epochs} epochs...')
    # Train the model
    best_f1 = train_model(model, train_loader, valid_loader, epochs=num_epochs, lr=0.001)
    print(f'Best F1 Score for fold {fold+1}: {best_f1:.4f}')

    # Save the model if needed
    torch.save(model.state_dict(), f'model_fold_{fold+1}.pth')

    # Save fold results
    fold_results[fold] = best_f1

    print('--------------------------------')

# Print fold results
print('K-FOLD CROSS VALIDATION RESULTS FOR {} FOLDS'.format(k_folds))
print('--------------------------------')
sum_f1 = 0.0
for key, value in fold_results.items():
    print(f'Fold {key+1}: Best F1 Score = {value:.4f}')
    sum_f1 += value
print('--------------------------------')
print(f'Average Best F1 Score: {sum_f1/len(fold_results):.4f}')

def load_model(model_path, num_classes):
    model = PretrainedResNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
