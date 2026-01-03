import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
from datetime import datetime


class RSSCN7Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.data_dir):
            raise ValueError(f"file not exist: {self.data_dir}")

        image_files = []
        for file in os.listdir(self.data_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                image_files.append(file)



        class_labels = set()
        for filename in image_files:

            if len(filename) > 0 and filename[0].isalpha():
                class_label = filename[0].lower()
                class_labels.add(class_label)


        self.classes = sorted(list(class_labels))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}



        for filename in image_files:
            if len(filename) > 0 and filename[0].isalpha():
                class_label = filename[0].lower()
                if class_label in self.class_to_idx:
                    class_idx = self.class_to_idx[class_label]
                    image_path = os.path.join(self.data_dir, filename)
                    self.samples.append((image_path, class_idx, filename))




        class_counts = defaultdict(int)
        for _, label, _ in self.samples:

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, filename = self.samples[idx]


        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (256, 256), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def create_data_loaders(data_dir, batch_size=32, train_split=0.7):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    full_dataset = RSSCN7Dataset(data_dir, transform=None)


    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size


    train_indices = []
    val_indices = []

    class_indices = defaultdict(list)
    for idx, (_, label, _) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)

    torch.manual_seed(42) 
    for class_label, indices in class_indices.items():

        indices = torch.tensor(indices)[torch.randperm(len(indices))].tolist()


        class_train_size = int(len(indices) * train_split)

        train_indices.extend(indices[:class_train_size])
        val_indices.extend(indices[class_train_size:])

        class_name = full_dataset.classes[class_label]
        print(f"  类别 '{class_name}': 训练 {class_train_size}, 验证 {len(indices) - class_train_size}")


    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)


    full_dataset.transform = train_transform
    train_dataset.dataset.transform = train_transform


    val_full_dataset = RSSCN7Dataset(data_dir, transform=val_transform)
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices)


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, full_dataset.classes


def create_model(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def save_model(model, classes, best_val_acc, save_dir, epoch, optimizer=None):

    os.makedirs(save_dir, exist_ok=True)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    model_filename = f"resnet18_rsscn7_best_acc{best_val_acc:.2f}_{timestamp}.pth"
    model_path = os.path.join(save_dir, model_filename)


    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'classes': classes,
        'class_to_idx': {class_name: idx for idx, class_name in enumerate(classes)},
        'num_classes': len(classes),
        'best_val_acc': best_val_acc,
        'model_architecture': 'resnet18',
        'dataset': 'RSSCN7',
        'timestamp': timestamp
    }

    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(save_dict, model_path)


    latest_path = os.path.join(save_dir, "resnet18_rsscn7_latest.pth")
    torch.save(save_dict, latest_path)

    return model_path


def train_model(model, train_loader, val_loader, classes, save_dir, num_epochs=50, learning_rate=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)


    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = None



    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            if batch_idx % 5 == 0: 
                print(f'  Batch {batch_idx:3d}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Acc: {100.0 * train_correct / train_total:.2f}%', end='\r')

        print()  


        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()


        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f'train - Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%')
        print(f'valid - Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_path = save_model(model, classes, best_val_acc, save_dir, epoch, optimizer)


        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']


    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'best_model_path': best_model_path
    }


def plot_training_history(history, save_dir):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    epochs = range(1, len(history['train_losses']) + 1)


    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss - RSSCN7', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)


    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy - RSSCN7', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)


    best_epoch = history['best_epoch']
    best_acc = history['best_val_acc']
    ax2.scatter(best_epoch, best_acc, color='red', s=100, zorder=5)
    ax2.annotate(f'Best: {best_acc:.2f}%\n(Epoch {best_epoch})',
                 xy=(best_epoch, best_acc), xytext=(10, 10),
                 textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()


    plot_path = os.path.join(save_dir, 'training_history_rsscn7.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图像已保存: {plot_path}")

    plt.show()


def load_model(model_path, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    num_classes = checkpoint['num_classes']
    model = create_model(num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)


    return model, checkpoint['classes'], checkpoint


def main():
    DATA_DIR = "/data2/lrf/data/RSSCN7-master/Total_GT"
    SAVE_DIR = "/data2/lrf/HIIF/models/cls/results"
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.7  # 70% 训练，30% 验证



    try:
        train_loader, val_loader, classes = create_data_loaders(
            DATA_DIR, BATCH_SIZE, TRAIN_SPLIT
        )
    except Exception as e:
        return

    model = create_model(num_classes=len(classes), pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


    model, history = train_model(
        model, train_loader, val_loader, classes, SAVE_DIR, NUM_EPOCHS, LEARNING_RATE
    )


    plot_training_history(history, SAVE_DIR)

    classes_info_path = os.path.join(SAVE_DIR, "rsscn7_class_info.txt")
    with open(classes_info_path, 'w') as f:
        f.write("RSSCN7 Remote Sensing Scene Classification - Class Information\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total classes: {len(classes)}\n")
        f.write(f"Best validation accuracy: {history['best_val_acc']:.2f}%\n")
        f.write(f"Best model epoch: {history['best_epoch']}\n")
        f.write(f"Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Class mapping:\n")
        for i, class_name in enumerate(classes):
            f.write(f"{i:2d}: '{class_name}'\n")
        f.write(f"\nDataset path: {DATA_DIR}\n")
        f.write(f"Model save path: {SAVE_DIR}\n")




if __name__ == "__main__":

    main()
