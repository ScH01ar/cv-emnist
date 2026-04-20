from __future__ import annotations  # 允许在类定义中延迟解析类型注解
import torch  # 导入 PyTorch 深度学习基础库
from torch import nn  # 导入神经网络模块，包含核心网络层组件

# 定义辅助函数：根据名称字符串返回相应的激活函数实例
def build_activation(name: str) -> nn.Module:
    key = name.lower()  # 将输入名称转为小写以实现不区分大小写的匹配
    if key == "relu": return nn.ReLU()  # 返回 ReLU 激活层
    if key == "leaky_relu": return nn.LeakyReLU(negative_slope=0.1)  # 返回 Leaky ReLU 激活层
    if key == "elu": return nn.ELU()  # 返回 ELU 激活层
    if key == "gelu": return nn.GELU()  # 返回 GELU 激活层
    if key == "silu": return nn.SiLU()  # 返回 SiLU (Swish) 激活层
    raise ValueError(f"Unsupported activation: {name}")  # 若不在支持列表中则抛出异常

# 定义辅助函数：根据名称字符串构建 2D 归一化层
def build_norm2d(name: str, num_features: int) -> nn.Module | None:
    key = name.lower()  # 名称转小写处理
    if key == "none": return None  # 如果设为 none，则返回空，表示不使用归一化
    if key == "batchnorm": return nn.BatchNorm2d(num_features)  # 返回针对卷积输出的 2D 批归一化层
    raise ValueError(f"Unsupported/Unimplemented normalization for 2D: {name}")  # 抛出不支持的异常

# 定义残差块 (Residual Block)，它是 ResNet 的核心构建单元
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation="relu", norm="batchnorm"):
        super().__init__()  # 调用基类初始化方法
        # 路径第一层：3x3 卷积，stride 参数决定了是否进行下采样
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 第一层卷积后的归一化，若未配置则设为 Identity
        self.bn1 = build_norm2d(norm, out_channels) if build_norm2d(norm, out_channels) else nn.Identity()
        # 加载指定的激活函数
        self.act = build_activation(activation)
        
        # 路径第二层：3x3 卷积，保持尺寸不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 第二层卷积后的归一化
        self.bn2 = build_norm2d(norm, out_channels) if build_norm2d(norm, out_channels) else nn.Identity()
        
        # 定义捷径连接 (Shortcut/Skip Connection)，用于将输入直接加到输出
        self.shortcut = nn.Sequential()
        # 如果步长不为 1 或输入输出通道不一致，则需要通过 1x1 卷积调整维度以实现相加
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                build_norm2d(norm, out_channels) if build_norm2d(norm, out_channels) else nn.Identity()
            )

    def forward(self, x):
        # 主路径计算流：卷积 -> 归一化 -> 激活 -> 卷积 -> 归一化
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 将主路径输出与捷径路径输出相加，实现残差连接
        out += self.shortcut(x) 
        # 最后再次经过激活函数
        out = self.act(out)
        return out

# 定义小型 ResNet 模型
class MiniResNet(nn.Module):
    def __init__(self, num_classes=47, activation="relu", norm="batchnorm", dropout=0.2):
        super().__init__()  # 基类初始化
        self.in_channels = 32  # 初始化当前层的输入通道数为 32
        
        # 初始处理层：直接对 1 通道输入图像进行特征初步提取
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = build_norm2d(norm, 32) if build_norm2d(norm, 32) else nn.Identity()
        self.act = build_activation(activation)
        
        # 构建第一个残差层：包含 2 个残差块，保持 32 通道，尺寸不变 (28x28)
        self.layer1 = self._make_layer(32, 2, stride=1, activation=activation, norm=norm)
        # 构建第二个残差层：包含 2 个残差块，增加到 64 通道，尺寸减半 (降采样到 14x14)
        self.layer2 = self._make_layer(64, 2, stride=2, activation=activation, norm=norm) 
        
        # 全局自适应平均池化，将特征图压缩为 1x1 尺寸
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 最终线性分类器
        self.fc = nn.Sequential(
            # 应用 Dropout 以增强泛化能力
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            # 全连接层将特征维度映射到最终的 47 个类别
            nn.Linear(64, num_classes)
        )

    # 内部方法：用于快速构建包含多个残差块的层级
    def _make_layer(self, out_channels, num_blocks, stride, activation, norm):
        strides = [stride] + [1]*(num_blocks-1)  # 仅首个块使用给定的步长进行下采样
        layers = []
        for s in strides:
            # 依次添加残差块，并更新通道计数
            layers.append(ResidualBlock(self.in_channels, out_channels, s, activation, norm))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始层处理
        x = self.act(self.bn1(self.conv1(x)))
        # 依次通过各个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        # 空间池化
        x = self.pool(x)
        # 将四维张量展平为二维以便进入全连接层
        x = torch.flatten(x, 1)
        # 分类器输出预测分数
        x = self.fc(x)
        return x