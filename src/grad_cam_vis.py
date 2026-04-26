import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def visualize_cnn_gradcam(model, target_layer, input_tensor, model_name):
    """
    CNN/ResNet 的特征图解释
    target_layer 示例: [model.layer2[-1].conv2] (ResNet)
    """
    # 1. 生成热力图
    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    
    # 2. 处理原图以符合 Grad-CAM 要求 [0, 1] 和 float32
    rgb_img = input_tensor[0, 0].cpu().numpy()
    
    # 【修复核心点】反归一化：假设你的 normalize 是 mean=0.5, std=0.5
    # 如果你没有用 Normalize，可以直接去掉这一行
    rgb_img = (rgb_img * 0.5) + 0.5 
    
    # 限制范围并强转格式
    rgb_img = np.clip(rgb_img, 0, 1)        # 强制截断在 0 到 1 之间
    rgb_img = np.float32(rgb_img)           # 强制转为 float32
    
    # 转换为 3 通道 (RGB) 格式
    rgb_img = np.stack((rgb_img,)*3, axis=-1)
    
    # 3. 叠加并保存
    vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    plt.figure()
    plt.imshow(vis)
    plt.title(f"{model_name} Grad-CAM")
    plt.axis('off')
    plt.savefig(f'results/{model_name}_gradcam.png', bbox_inches='tight')
    plt.close()

def visualize_vit_attention(model, input_tensor):
    """
    ViT 注意力权重热力图 (Attention Rollout)
    """
    model.eval()
    # 假设你的 ViT 实现中可以访问注意力权重，或者通过 Hook 提取
    # 这里演示一种通用的思路：提取最后一个 Block 的 Attention
    attentions = model.get_last_self_attention(input_tensor) # 需要在模型类中实现此方法
    
    # 均值化多头注意力
    nh = attentions.shape[1] # number of heads
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1) # 只取 CLS token 对 patch 的注意力
    
    # 将 1D 序列恢复为 2D 空间热力图 (例如 28x28 图像下 patch_size=7, 则为 4x4)
    w_featmap = input_tensor.shape[-2] // model.patch_size
    h_featmap = input_tensor.shape[-1] // model.patch_size
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    avg_attention = torch.mean(attentions, dim=0).cpu().numpy()
    
    plt.imshow(avg_attention, cmap='inferno')
    plt.title("ViT Attention Map")
    plt.savefig('results/vit_attention.png')