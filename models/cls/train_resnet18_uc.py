import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from datetime import datetime


class UCMercedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        UCMerced数据集加载器
        Args:
            data_dir: 数据集根目录
            transform: 数据预处理变换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        # 扫描数据目录，提取图片和标签
        self._load_dataset()

    def _load_dataset(self):
        """加载数据集，从文件名提取标签"""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据目录不存在: {self.data_dir}")

        # 收集所有图片文件
        image_files = []
        for file in os.listdir(self.data_dir):
            if file.lower().endswith(('.tif', '.jpg', '.jpeg', '.png')):
                image_files.append(file)

        if len(image_files) == 0:
            raise ValueError(f"在目录 {self.data_dir} 中没有找到图片文件")

        # 从文件名提取类别标签
        class_names = set()
        for filename in image_files:
            # 提取类别名（去掉数字和扩展名）
            # 例如: agricultural00.tif -> agricultural
            class_name = re.sub(r'\d+\..*$', '', filename)
            class_names.add(class_name)

        # 创建类别到索引的映射
        self.classes = sorted(list(class_names))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        print(f"发现 {len(self.classes)} 个类别: {self.classes}")

        # 创建样本列表
        for filename in image_files:
            class_name = re.sub(r'\d+\..*$', '', filename)
            class_idx = self.class_to_idx[class_name]
            image_path = os.path.join(self.data_dir, filename)
            self.samples.append((image_path, class_idx))

        print(f"总共加载了 {len(self.samples)} 个样本")

        # 统计每个类别的样本数
        class_counts = defaultdict(int)
        for _, label in self.samples:
            class_counts[self.classes[label]] += 1

        print("各类别样本数统计:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        # 加载图片
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图片 {image_path}: {e}")
            # 返回一个默认的黑色图片
            image = Image.new('RGB', (256, 256), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def create_data_loaders(data_dir, batch_size=32, train_split=0.7):
    """创建训练和验证数据加载器 (70% 训练, 30% 验证)"""

    # 数据预处理
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

    # 加载完整数据集
    full_dataset = UCMercedDataset(data_dir, transform=None)

    # 分割数据集 (70% 训练, 30% 验证)
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    print(f"\n数据集分割:")
    print(f"训练集: {train_size} 样本 ({train_split * 100:.0f}%)")
    print(f"验证集: {val_size} 样本 ({(1 - train_split) * 100:.0f}%)")

    # 随机分割
    torch.manual_seed(42)  # 确保可重复性
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 设置不同的变换
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # 创建数据加载器
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
    """创建ResNet18模型"""
    model = models.resnet18(pretrained=pretrained)

    # 修改最后一层以适应类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def save_model(model, classes, best_val_acc, save_dir, epoch, optimizer=None):
    """保存模型到指定路径"""
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 模型文件名
    model_filename = f"resnet18_ucmerced_best_acc{best_val_acc:.2f}_{timestamp}.pth"
    model_path = os.path.join(save_dir, model_filename)

    # 保存模型和相关信息
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'classes': classes,
        'class_to_idx': {class_name: idx for idx, class_name in enumerate(classes)},
        'num_classes': len(classes),
        'best_val_acc': best_val_acc,
        'model_architecture': 'resnet18',
        'timestamp': timestamp
    }

    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(save_dict, model_path)

    # 同时保存一个最新的模型（便于加载）
    latest_path = os.path.join(save_dir, "resnet18_ucmerced_latest.pth")
    torch.save(save_dict, latest_path)

    return model_path


def train_model(model, train_loader, val_loader, classes, save_dir, num_epochs=50, learning_rate=0.001):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 训练历史记录
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = None

    print(f"\n开始训练，模型将保存到: {save_dir}")
    print("=" * 80)

    for epoch in range(num_epochs):
        # 训练阶段
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

            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx:3d}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Acc: {100.0 * train_correct / train_total:.2f}%', end='\r')

        print()  # 换行

        # 验证阶段
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

        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f'训练 - Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%')
        print(f'验证 - Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_path = save_model(model, classes, best_val_acc, save_dir, epoch, optimizer)
            print(
                f'🎉 新的最佳模型! 验证准确率: {best_val_acc:.2f}% (提升 {val_acc - best_val_acc + (val_acc - best_val_acc):.2f}%)')
            print(f'   模型已保存: {best_model_path}')

        print(f'当前最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})')

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'学习率: {current_lr:.6f}')
        print("=" * 80)

    print(f"\n✅ 训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"最佳模型保存路径: {best_model_path}")

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
    """绘制并保存训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    epochs = range(1, len(history['train_losses']) + 1)

    # 损失曲线
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 标注最佳准确率点
    best_epoch = history['best_epoch']
    best_acc = history['best_val_acc']
    ax2.scatter(best_epoch, best_acc, color='red', s=100, zorder=5)
    ax2.annotate(f'Best: {best_acc:.2f}%\n(Epoch {best_epoch})',
                 xy=(best_epoch, best_acc), xytext=(10, 10),
                 textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    # 保存图像
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图像已保存: {plot_path}")

    plt.show()


def load_model(model_path, device=None):
    """加载保存的模型"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    # 重建模型
    num_classes = checkpoint['num_classes']
    model = create_model(num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"模型加载成功!")
    print(f"  架构: {checkpoint['model_architecture']}")
    print(f"  类别数: {num_classes}")
    print(f"  最佳验证准确率: {checkpoint['best_val_acc']:.2f}%")
    print(f"  训练时间: {checkpoint['timestamp']}")

    return model, checkpoint['classes'], checkpoint


def main():
    # 配置参数
    DATA_DIR = "/data2/lrf/data/UCMerced_LandUse/Images/Total_GT"
    SAVE_DIR = "/data2/lrf/HIIF/models/cls/results"
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.7  # 70% 训练，30% 验证

    print("🚀 开始UCMerced土地利用分类训练")
    print("=" * 80)
    print(f"数据目录: {DATA_DIR}")
    print(f"模型保存目录: {SAVE_DIR}")
    print(f"训练/验证比例: {int(TRAIN_SPLIT * 100)}% / {int((1 - TRAIN_SPLIT) * 100)}%")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")

    # 创建数据加载器
    try:
        train_loader, val_loader, classes = create_data_loaders(
            DATA_DIR, BATCH_SIZE, TRAIN_SPLIT
        )
        print(f"\n✅ 数据集加载成功！")
        print(f"类别数: {len(classes)}")
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    # 创建模型
    print(f"\n🔧 创建ResNet18模型（使用ImageNet预训练权重）...")
    model = create_model(num_classes=len(classes), pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 训练模型
    print(f"\n🏋️ 开始训练模型...")
    model, history = train_model(
        model, train_loader, val_loader, classes, SAVE_DIR, NUM_EPOCHS, LEARNING_RATE
    )

    # 绘制训练历史
    plot_training_history(history, SAVE_DIR)

    # 保存类别信息
    classes_info_path = os.path.join(SAVE_DIR, "class_info.txt")
    with open(classes_info_path, 'w') as f:
        f.write("UCMerced Land Use Classification - Class Information\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total classes: {len(classes)}\n")
        f.write(f"Best validation accuracy: {history['best_val_acc']:.2f}%\n")
        f.write(f"Best model epoch: {history['best_epoch']}\n\n")
        f.write("Class mapping:\n")
        for i, class_name in enumerate(classes):
            f.write(f"{i:2d}: {class_name}\n")

    print(f"\n📝 类别信息已保存: {classes_info_path}")
    print(f"📊 训练历史图像已保存: {os.path.join(SAVE_DIR, 'training_history.png')}")

    print("\n" + "=" * 80)
    print("🎉 训练任务完成!")
    print(f"最佳验证准确率: {history['best_val_acc']:.2f}%")
    print(f"最佳模型路径: {history['best_model_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()