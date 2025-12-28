import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
from collections import defaultdict
import pandas as pd


class UCMercedTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        UCMerced测试数据集加载器
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
        """加载测试数据集，从文件名提取标签"""
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

        print(f"测试集发现 {len(self.classes)} 个类别: {self.classes}")

        # 创建样本列表
        for filename in image_files:
            class_name = re.sub(r'\d+\..*$', '', filename)
            if class_name in self.class_to_idx:  # 确保类别存在
                class_idx = self.class_to_idx[class_name]
                image_path = os.path.join(self.data_dir, filename)
                self.samples.append((image_path, class_idx, filename))

        print(f"总共加载了 {len(self.samples)} 个测试样本")

        # 统计每个类别的样本数
        class_counts = defaultdict(int)
        for _, label, _ in self.samples:
            class_counts[self.classes[label]] += 1

        print("各类别测试样本数统计:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, filename = self.samples[idx]

        # 加载图片
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图片 {image_path}: {e}")
            # 返回一个默认的黑色图片
            image = Image.new('RGB', (256, 256), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label, filename


def create_model(num_classes):
    """创建ResNet18模型架构"""
    model = models.resnet18(pretrained=False)  # 不需要预训练权重，会加载我们的权重
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(model_path, device=None):
    """加载保存的模型"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # 重建模型
    num_classes = checkpoint['num_classes']
    model = create_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # 设置为评估模式

    print(f"✅ 模型加载成功!")
    print(f"  架构: {checkpoint.get('model_architecture', 'resnet18')}")
    print(f"  类别数: {num_classes}")
    print(f"  训练时最佳验证准确率: {checkpoint['best_val_acc']:.2f}%")
    print(f"  训练时间: {checkpoint.get('timestamp', 'Unknown')}")
    print(f"  训练轮数: {checkpoint.get('epoch', 'Unknown')}")

    return model, checkpoint['classes'], checkpoint


def create_test_dataloader(data_dir, batch_size=32):
    """创建测试数据加载器"""
    # 测试时的数据预处理（不包含数据增强）
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = UCMercedTestDataset(data_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return test_loader, test_dataset.classes


def test_model(model, test_loader, device, trained_classes):
    """测试模型并收集详细结果"""
    model.eval()

    all_predictions = []
    all_labels = []
    all_filenames = []
    all_probabilities = []

    correct = 0
    total = 0

    print("开始测试...")
    print("-" * 60)

    with torch.no_grad():
        for batch_idx, (data, target, filenames) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            # 前向传播
            outputs = model(data)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # 收集结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_filenames.extend(filenames)
            all_probabilities.extend(probabilities.cpu().numpy())

            # 计算准确率
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if (batch_idx + 1) % 10 == 0:
                current_acc = 100.0 * correct / total
                print(f"已处理 {total} 个样本，当前准确率: {current_acc:.2f}%")

    # 计算最终准确率
    final_accuracy = 100.0 * correct / total
    print(f"\n✅ 测试完成！")
    print(f"总样本数: {total}")
    print(f"正确预测: {correct}")
    print(f"测试准确率: {final_accuracy:.2f}%")

    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'filenames': all_filenames,
        'probabilities': all_probabilities,
        'accuracy': final_accuracy,
        'total_samples': total,
        'correct_samples': correct
    }


def calculate_metrics(results, class_names):
    """计算各种评估指标"""
    y_true = results['labels']
    y_pred = results['predictions']

    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)

    # 计算每个类别的精确率、召回率、F1分数
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )

    # 计算宏平均和微平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro'
    )

    # 计算加权平均
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm
    }


def save_detailed_results(results, metrics, class_names, model_info, save_dir):
    """保存详细的测试结果到txt文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"test_results_{timestamp}.txt"
    results_path = os.path.join(save_dir, results_filename)

    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("UCMerced Land Use Classification - Test Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型文件: {model_info.get('model_path', 'Unknown')}\n")
        f.write(f"训练时最佳验证准确率: {model_info.get('best_val_acc', 'Unknown'):.2f}%\n")
        f.write(f"训练轮数: {model_info.get('epoch', 'Unknown')}\n")
        f.write(f"训练时间: {model_info.get('timestamp', 'Unknown')}\n")
        f.write("\n" + "=" * 80 + "\n")

        # 总体结果
        f.write("总体测试结果:\n")
        f.write("-" * 40 + "\n")
        f.write(f"总测试样本数: {results['total_samples']}\n")
        f.write(f"正确预测数: {results['correct_samples']}\n")
        f.write(f"测试准确率: {results['accuracy']:.4f} ({results['accuracy']:.2f}%)\n")
        f.write("\n")

        # 平均指标
        f.write("平均指标:\n")
        f.write("-" * 40 + "\n")
        f.write(f"精确率 (Macro Average): {metrics['precision_macro']:.4f}\n")
        f.write(f"召回率 (Macro Average): {metrics['recall_macro']:.4f}\n")
        f.write(f"F1分数 (Macro Average): {metrics['f1_macro']:.4f}\n")
        f.write(f"精确率 (Micro Average): {metrics['precision_micro']:.4f}\n")
        f.write(f"召回率 (Micro Average): {metrics['recall_micro']:.4f}\n")
        f.write(f"F1分数 (Micro Average): {metrics['f1_micro']:.4f}\n")
        f.write(f"精确率 (Weighted Average): {metrics['precision_weighted']:.4f}\n")
        f.write(f"召回率 (Weighted Average): {metrics['recall_weighted']:.4f}\n")
        f.write(f"F1分数 (Weighted Average): {metrics['f1_weighted']:.4f}\n")
        f.write("\n")

        # 每个类别的详细指标
        f.write("各类别详细指标:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<10} {'准确率':<10}\n")
        f.write("-" * 80 + "\n")

        for i, class_name in enumerate(class_names):
            # 计算该类别的准确率
            class_correct = sum(1 for true_label, pred_label in zip(results['labels'], results['predictions'])
                                if true_label == i and pred_label == i)
            class_total = sum(1 for label in results['labels'] if label == i)
            class_accuracy = class_correct / class_total if class_total > 0 else 0

            f.write(f"{class_name:<15} {metrics['precision_per_class'][i]:<10.4f} "
                    f"{metrics['recall_per_class'][i]:<10.4f} {metrics['f1_per_class'][i]:<10.4f} "
                    f"{metrics['support_per_class'][i]:<10} {class_accuracy:<10.4f}\n")

        f.write("\n")

        # 混淆矩阵
        f.write("混淆矩阵 (行:真实标签, 列:预测标签):\n")
        f.write("-" * 80 + "\n")

        # 表头
        f.write(f"{'类别':<12}")
        for class_name in class_names:
            f.write(f"{class_name[:8]:<8}")
        f.write("\n")
        f.write("-" * (12 + 8 * len(class_names)) + "\n")

        # 混淆矩阵数据
        cm = metrics['confusion_matrix']
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name[:12]:<12}")
            for j in range(len(class_names)):
                f.write(f"{cm[i][j]:<8}")
            f.write("\n")

        f.write("\n")

        # 分类报告
        f.write("Sklearn分类报告:\n")
        f.write("-" * 80 + "\n")
        report = classification_report(results['labels'], results['predictions'],
                                       target_names=class_names, digits=4)
        f.write(report)
        f.write("\n")

        # 预测错误的样本
        f.write("预测错误的样本 (仅显示前50个):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'文件名':<25} {'真实标签':<15} {'预测标签':<15} {'置信度':<10}\n")
        f.write("-" * 80 + "\n")

        error_count = 0
        for i, (filename, true_label, pred_label, prob) in enumerate(
                zip(results['filenames'], results['labels'], results['predictions'], results['probabilities'])
        ):
            if true_label != pred_label and error_count < 50:
                confidence = prob[pred_label]
                f.write(f"{filename[:25]:<25} {class_names[true_label]:<15} "
                        f"{class_names[pred_label]:<15} {confidence:<10.4f}\n")
                error_count += 1

        if error_count == 50:
            total_errors = sum(1 for true_label, pred_label in zip(results['labels'], results['predictions'])
                               if true_label != pred_label)
            f.write(f"... (总共 {total_errors} 个错误预测，仅显示前50个)\n")

    print(f"📝 详细测试结果已保存到: {results_path}")
    return results_path


def save_confusion_matrix_plot(confusion_matrix, class_names, save_dir):
    """保存混淆矩阵可视化图"""
    plt.figure(figsize=(12, 10))

    # 使用热力图显示混淆矩阵
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix - UCMerced Land Use Classification', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📊 混淆矩阵图已保存到: {plot_path}")
    return plot_path


def save_results_csv(results, class_names, save_dir):
    """保存详细结果到CSV文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f"detailed_predictions_{timestamp}.csv")

    # 创建DataFrame
    data = []
    for i, (filename, true_label, pred_label, prob) in enumerate(
            zip(results['filenames'], results['labels'], results['predictions'], results['probabilities'])
    ):
        row = {
            'filename': filename,
            'true_class': class_names[true_label],
            'predicted_class': class_names[pred_label],
            'correct': true_label == pred_label,
            'confidence': prob[pred_label],
            'true_class_prob': prob[true_label]
        }

        # 添加所有类别的概率
        for j, class_name in enumerate(class_names):
            row[f'prob_{class_name}'] = prob[j]

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"📊 详细预测结果CSV已保存到: {csv_path}")
    return csv_path


def main():
    # 配置参数
    MODEL_PATH = "/data2/lrf/HIIF/models/cls/results/resnet18_ucmerced_best.pth"
    TEST_DATA_DIR = "/data2/lrf/IDM/experiments/uc_x8"
    SAVE_DIR = "/data2/lrf/HIIF/models/cls/results/idm"
    BATCH_SIZE = 32

    print("🧪 UCMerced土地利用分类 - 模型测试")
    print("=" * 80)
    print(f"模型路径: {MODEL_PATH}")
    print(f"测试数据路径: {TEST_DATA_DIR}")
    print(f"结果保存路径: {SAVE_DIR}")
    print(f"批次大小: {BATCH_SIZE}")

    # 检查路径是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return

    if not os.path.exists(TEST_DATA_DIR):
        print(f"❌ 测试数据目录不存在: {TEST_DATA_DIR}")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        # 加载模型
        print(f"\n📥 加载训练好的模型...")
        model, trained_classes, checkpoint = load_model(MODEL_PATH, device)

        # 创建测试数据加载器
        print(f"\n📂 加载测试数据...")
        test_loader, test_classes = create_test_dataloader(TEST_DATA_DIR, BATCH_SIZE)

        # 检查类别是否匹配
        print(f"\n🔍 检查类别匹配性...")
        print(f"训练时的类别数: {len(trained_classes)}")
        print(f"测试数据的类别数: {len(test_classes)}")

        if set(trained_classes) != set(test_classes):
            print("⚠️  警告: 训练和测试的类别不完全匹配!")
            print(f"训练类别: {sorted(trained_classes)}")
            print(f"测试类别: {sorted(test_classes)}")
        else:
            print("✅ 训练和测试类别完全匹配!")

        # 使用训练时的类别顺序
        class_names = trained_classes

        # 进行测试
        print(f"\n🧪 开始模型测试...")
        results = test_model(model, test_loader, device, trained_classes)

        # 计算评估指标
        print(f"\n📊 计算评估指标...")
        metrics = calculate_metrics(results, class_names)

        # 准备模型信息
        model_info = {
            'model_path': MODEL_PATH,
            'best_val_acc': checkpoint.get('best_val_acc', 0),
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'timestamp': checkpoint.get('timestamp', 'Unknown')
        }

        # 保存详细结果
        print(f"\n💾 保存测试结果...")
        results_path = save_detailed_results(results, metrics, class_names, model_info, SAVE_DIR)

        # 保存混淆矩阵图
        confusion_matrix_path = save_confusion_matrix_plot(metrics['confusion_matrix'], class_names, SAVE_DIR)

        # 保存CSV结果
        csv_path = save_results_csv(results, class_names, SAVE_DIR)

        # 打印总结
        print(f"\n" + "=" * 80)
        print(f"🎉 测试完成! 结果总结:")
        print(f"📈 测试准确率: {results['accuracy']:.2f}%")
        print(f"📊 F1分数 (宏平均): {metrics['f1_macro']:.4f}")
        print(f"📊 F1分数 (加权平均): {metrics['f1_weighted']:.4f}")
        print(f"📝 详细结果: {results_path}")
        print(f"📊 混淆矩阵: {confusion_matrix_path}")
        print(f"📋 CSV文件: {csv_path}")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()