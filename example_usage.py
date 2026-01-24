import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import os

# 导入我们的模型
from ZR2ViM import ZR2ViM_Seg

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建模型
def create_model(img_size=224, in_chans=3, num_classes=1):
    model = ZR2ViM_Seg(
        img_size=img_size,
        patch_size=4,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        decoder_depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.1,
    )
    return model

# 示例：如何使用模型进行前向传播
def test_forward_pass():
    # 创建一个224x224的3通道随机图像
    batch_size = 2
    img_size = 224
    in_chans = 3
    num_classes = 1
    
    # 创建随机输入
    x = torch.randn(batch_size, in_chans, img_size, img_size)
    
    # 创建模型
    model = create_model(img_size, in_chans, num_classes)
    model.eval()
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 验证输出形状是否正确
    expected_shape = (batch_size, num_classes, img_size, img_size)
    assert output.shape == expected_shape, f"输出形状 {output.shape} 与预期 {expected_shape} 不符"
    print("前向传播测试通过！")

# 示例：训练函数
def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:  # 每10个批次打印一次
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {running_loss/10:.4f}')
                running_loss = 0.0

# 示例：评估函数
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 对于分割任务，可以计算IoU或Dice系数
            # 这里简化为二分类准确率
            predicted = (outputs > 0.5).float()
            total += targets.size(0) * targets.size(2) * targets.size(3)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f'测试准确率: {accuracy:.2f}%')
    return accuracy

# 主函数
def main():
    # 测试前向传播
    test_forward_pass()
    
    # 在实际应用中，您需要准备数据集并进行训练
    # 这里仅作为示例，展示如何使用模型
    print("\n模型已成功创建并通过前向传播测试！")
    print("在实际应用中，您需要准备数据集并进行训练。")

if __name__ == "__main__":
    main()
