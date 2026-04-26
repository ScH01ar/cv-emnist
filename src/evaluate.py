import sys
import os
# 将项目的根目录强制加入 Python 的搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
# 请确保正确导入你的模型类和数据集构建函数
from models.cnn import CNN
from models.resnet import MiniResNet
from models.mlp import MLP
# from models.vit import VisionTransformer
from dataset import build_dataloaders

def evaluate_model_performance(model, dataloader, device, class_names, model_name):
    """
    实现 6.a.ii (混淆矩阵) 和 6.a.iii (四项指标汇总)
    """
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    # 1. 计算指标 (6.a.iii)
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    metrics = {
        "Model": model_name,
        "Accuracy": report["accuracy"],
        "Precision": report["macro avg"]["precision"],
        "Recall": report["macro avg"]["recall"],
        "F1-Score": report["macro avg"]["f1-score"]
    }
    
    # 2. 绘制混淆矩阵 (6.a.ii)
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.savefig(f'results/{model_name}_cm.png')
    plt.close()
    
    return metrics

def plot_top_6_predictions(models_dict, test_loader, device, class_names):
    """
    实现 6.a.i: 输出测试集中前六个样本的预测结果对比
    """
    inputs, targets = next(iter(test_loader))
    inputs, targets = inputs[:6].to(device), targets[:6].to(device)
    
    n_models = len(models_dict)
    fig, axes = plt.subplots(n_models + 1, 6, figsize=(15, 3 * (n_models + 1)))
    
    # 第一行显示原图
    for i in range(6):
        img = inputs[i][0].cpu().numpy() * 0.5 + 0.5 # 反归一化
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f"True: {class_names[targets[i]]}")
        axes[0, i].axis('off')

    # 后续行显示各模型的预测结果
    for row, (name, model) in enumerate(models_dict.items(), 1):
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
        for i in range(6):
            pred_label = class_names[preds[i]]
            color = "green" if preds[i] == targets[i] else "red"
            axes[row, i].text(0.5, 0.5, f"{pred_label}", fontsize=12, 
                             ha='center', va='center', color=color)
            axes[row, i].set_ylabel(name, rotation=0, labelpad=40)
            axes[row, i].axis('off')
            
    plt.tight_layout()
    plt.savefig('results/top_6_comparison.png')

def main():
    # 1. 基础设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估设备: {device}")
    
    # 创建保存结果的文件夹
    os.makedirs('results', exist_ok=True)

    # 2. 加载测试集数据 (假设你只需要 test_loader)
    # 注意：这里的 batch_size 可以设大一点加快评估速度
    _, _, test_loader, num_classes = build_dataloaders(batch_size=256)
    
    # 为了可视化好看，你可以定义一个 EMNIST 字符映射表
    # 如果没有映射表，可以直接用数字转字符串，例如 [str(i) for i in range(47)]
    class_names = [str(i) for i in range(num_classes)] 

    # 3. 实例化你的最优模型架构
    # 【注意】这里的参数必须和你训练跑出最高分时 YAML 里的 kwargs 完全一致！
    print("正在实例化模型架构...")
    mlp_model = MLP(activation='relu', norm='batchnorm', dropout=0.2).to(device)
    cnn_model = CNN(activation='gelu', norm='batchnorm', dropout=0.2).to(device)
    resnet_model = MiniResNet(activation='gelu', norm='batchnorm', dropout=0.0).to(device)
    
    # 4. 加载你训练好的最优权重 (best.pt)
    # 【修改这里】换成你实际保存 best.pt 的路径
    print("正在加载模型权重...")
    mlp_model.load_state_dict(torch.load('artifacts/best_runs/mlp/best.pt', map_location=device))
    cnn_model.load_state_dict(torch.load('artifacts/best_runs/cnn/best.pt', map_location=device))
    resnet_model.load_state_dict(torch.load('artifacts/best_runs/resnet/best.pt', map_location=device))

    # 将你的模型打包成字典，方便后续循环处理
    models_dict = {
        "CNN": cnn_model,
        "ResNet": resnet_model,
        "MLP": mlp_model,  
        # "ViT": vit_model
    }

    # ================= 开始执行考核要求 =================
    
    # [要求 6.a.ii & 6.a.iii]：计算指标并画混淆矩阵
    print("\n--- 开始计算各项指标 ---")
    all_metrics = []
    for name, model in models_dict.items():
        print(f"正在评估 {name} ...")
        metrics = evaluate_model_performance(model, test_loader, device, class_names, name)
        all_metrics.append(metrics)
        
    # 打印最终的总表
    print("\n=== 模型性能对比总表 ===")
    import pandas as pd
    df = pd.DataFrame(all_metrics)
    print(df.to_string(index=False))
    df.to_csv('results/model_comparison.csv', index=False)

    # [要求 6.a.i]：绘制前 6 个样本的预测对比
    print("\n--- 正在生成前6个样本对比图 ---")
    plot_top_6_predictions(models_dict, test_loader, device, class_names)
    
    # [要求 6.b]：可视化特征图 (以 CNN 为例)
    print("\n--- 正在生成 Grad-CAM 特征图 ---")
    try:
        from grad_cam_vis import visualize_cnn_gradcam
        # 取出一个样本
        inputs, _ = next(iter(test_loader))
        single_input = inputs[0:1].to(device) # 取第一张图片，保持 [1, C, H, W] 形状
        
        # 传入 CNN 进行可视化
        import torch.nn as nn
        print("\n--- 正在生成 CNN Grad-CAM 特征图 ---")
        # 1. 动态寻找 CNN 的最后一个卷积层
        target_layer = None
        for module in reversed(list(cnn_model.modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = [module]
                break
                
        # 2. 执行可视化
        if target_layer is not None:
            print(f"成功找到目标卷积层: {target_layer[0]}")
            visualize_cnn_gradcam(cnn_model, target_layer, single_input, "CNN")
            print("CNN Grad-CAM 生成成功！")
        else:
            print("警告: 在模型中没有找到任何 Conv2d 层，无法生成 Grad-CAM！")
        print("\n--- 正在生成 ResNet Grad-CAM 特征图 ---")
        resnet_target_layer = None
        for module in reversed(list(resnet_model.modules())):
            if isinstance(module, nn.Conv2d):
                resnet_target_layer = [module]
                break
                
        if resnet_target_layer is not None:
            visualize_cnn_gradcam(resnet_model, resnet_target_layer, single_input, "ResNet")
            print("ResNet Grad-CAM 生成成功！")
        print("Grad-CAM 生成成功，请在 results 文件夹查看。")
    except Exception as e:
        print(f"Grad-CAM 运行跳过或失败，原因: {e}")

    print("\n全部评估完成！所有图片和表格已保存在 results/ 目录下。")

if __name__ == '__main__':
    main()