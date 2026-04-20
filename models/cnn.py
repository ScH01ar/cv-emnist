from __future__ import annotations  # 导入类型提示增强功能，允许在类内部使用类名作为类型
import torch  # 导入 PyTorch 深度学习框架
from torch import nn  # 导入神经网络模块，用于构建网络层

# 定义一个辅助函数，根据配置字符串返回相应的激活函数实例
def build_activation(name: str) -> nn.Module:
    key = name.lower()  # 将名称统一转为小写，增强配置的兼容性
    if key == "relu": return nn.ReLU()  # 返回线性整流函数
    if key == "leaky_relu": return nn.LeakyReLU(negative_slope=0.1)  # 返回带负斜率的 ReLU
    if key == "elu": return nn.ELU()  # 返回指数线性单元
    if key == "gelu": return nn.GELU()  # 返回高斯误差线性单元
    if key == "silu": return nn.SiLU()  # 返回 Sigmoid 线性单元 (Swish)
    raise ValueError(f"Unsupported activation: {name}")  # 报错：不支持的激活函数

# 定义一个辅助函数，根据配置字符串返回相应的 2D 归一化层
def build_norm2d(name: str, num_features: int) -> nn.Module | None:
    key = name.lower()  # 将名称统一转为小写
    if key == "none": return None  # 如果配置为 none，则不使用归一化
    if key == "batchnorm": return nn.BatchNorm2d(num_features)  # 返回批归一化层，输入参数为通道数
    raise ValueError(f"Unsupported/Unimplemented normalization for 2D: {name}")  # 报错：不支持的归一化方式

# 定义用于 EMNIST 分类的卷积神经网络模型
class CNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 47,  # 任务分类总数，Balanced 子集为 47
        activation: str = "relu",  # 使用哪种激活函数
        norm: str = "none",  # 是否使用归一化
        dropout: float = 0.2,  # Dropout 丢弃率
    ) -> None:
        super().__init__()  # 初始化父类
        
        # 构建特征提取层 (至少包含两层卷积)
        self.features = nn.Sequential(
            # 第一层卷积：输入单通道图像，输出 32 个特征图，卷积核 3x3
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # 根据参数添加归一化层，如果没有则使用 Identity (原样输出)
            build_norm2d(norm, 32) if build_norm2d(norm, 32) else nn.Identity(),
            # 应用激活函数
            build_activation(activation),
            # 最大池化层，将尺寸从 28x28 压缩至 14x14
            nn.MaxPool2d(2, 2), 
            
            # 第二层卷积：输入 32 通道，输出 64 通道，卷积核 3x3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # 应用第二层的归一化
            build_norm2d(norm, 64) if build_norm2d(norm, 64) else nn.Identity(),
            # 应用激活函数
            build_activation(activation),
            # 最大池化层，将尺寸从 14x14 压缩至 7x7
            nn.MaxPool2d(2, 2), 
        )
        
        # 构建分类层 (全连接层)
        self.classifier = nn.Sequential(
            # 将卷积层输出的 64x7x7 三维张量展平为一维向量
            nn.Flatten(),
            # 隐藏层：将展平后的特征映射到 256 个神经元
            nn.Linear(64 * 7 * 7, 256),
            # 应用激活函数
            build_activation(activation),
            # 应用 Dropout 防止过拟合
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            # 输出层：将 256 个特征映射到 47 个类别得分
            nn.Linear(256, num_classes)
        )

    # 定义数据流向的前向传播过程
    def forward(self, x):
        x = self.features(x)  # 通过卷积特征提取层
        x = self.classifier(x)  # 通过全连接分类层
        return x  # 返回最终预测结果