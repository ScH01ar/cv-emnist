from __future__ import annotations  # 启用延迟类型注解，方便书写较复杂的类型标注。

from collections import OrderedDict  # 导入有序字典，用于固定模型加载与遍历顺序。
from pathlib import Path  # 导入 Path，统一处理文件系统路径。

import matplotlib.pyplot as plt  # 导入 matplotlib，用于绘制特征图、Grad-CAM 和注意力热力图。
import torch  # 导入 PyTorch，负责模型加载、推理和张量运算。
import torch.nn.functional as F  # 导入函数式接口，用于插值放大热力图等操作。

from dataset import build_dataloaders  # 导入数据集构建函数，用于读取测试集。
from utils import PROJECT_ROOT, ensure_dir, get_device, load_class_from_file, load_yaml, save_json, set_seed  # 导入项目工具函数。

BEST_RUNS_DIR = PROJECT_ROOT / "artifacts" / "best_runs"  # 指定最优模型权重与配置所在目录。
RESULTS_DIR = PROJECT_ROOT / "results" / "interpretability_analysis"  # 指定可解释性分析结果输出目录。
NORMALIZE_MEAN = 0.1307  # 记录归一化使用的均值，便于可视化时反归一化。
NORMALIZE_STD = 0.3081  # 记录归一化使用的标准差，便于可视化时反归一化。


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:  # 定义图像反归一化函数。
    return image_tensor * NORMALIZE_STD + NORMALIZE_MEAN  # 将标准化图像恢复到原始灰度范围附近。


def load_model_from_run(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:  # 根据 best_runs 中的配置和权重加载单个模型。
    config = load_yaml(run_dir / "config.yaml")  # 读取当前模型对应的 YAML 配置文件。
    model_config = config["model"]  # 取出模型定义相关配置。
    model_class = load_class_from_file(PROJECT_ROOT / model_config["file"], model_config["class_name"])  # 按文件路径和类名动态导入模型类。
    model = model_class(**model_config.get("kwargs", {}))  # 使用配置中的参数实例化模型。
    state_dict = torch.load(run_dir / "best.pt", map_location=device)  # 将最优权重加载到指定设备。
    model.load_state_dict(state_dict)  # 把权重写入模型实例。
    model.to(device)  # 将模型移动到 CPU 或 GPU。
    model.eval()  # 切换到评估模式。
    return model, config  # 返回模型对象和其配置内容。


def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:  # 查找 CNN 或 ResNet 中最后一个卷积层。
    for module in reversed(list(model.modules())):  # 倒序遍历所有子模块，以便尽快找到最后一层卷积。
        if isinstance(module, torch.nn.Conv2d):  # 如果当前子模块是二维卷积层。
            return module  # 立即返回该卷积层。
    raise ValueError(f"No Conv2d layer found in {model.__class__.__name__}.")  # 如果模型中不存在卷积层则报错。


@torch.no_grad()  # 关闭梯度计算，因为这里只需要前向预测和置信度。
def get_prediction(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device) -> tuple[int, float]:  # 获取单个样本的预测类别和对应置信度。
    logits = model(input_tensor.to(device))  # 将输入移动到指定设备并执行前向推理。
    probabilities = torch.softmax(logits, dim=1)  # 对 logits 做 softmax，转成类别概率。
    predicted_class = int(probabilities.argmax(dim=1).item())  # 取概率最大的类别作为预测结果。
    confidence = float(probabilities[0, predicted_class].item())  # 读取该预测类别对应的概率作为置信度。
    return predicted_class, confidence  # 返回预测类别编号与置信度。


def choose_interpretability_sample(models: OrderedDict[str, torch.nn.Module], dataloader, device: torch.device) -> tuple[torch.Tensor, int]:  # 从测试集中挑选一个被 CNN、ResNet 和 ViT 同时预测正确的样本。
    for inputs, targets in dataloader:  # 逐批遍历测试集。
        batch_predictions: dict[str, torch.Tensor] = {}  # 初始化字典，用于保存当前批次中各模型的预测结果。
        for model_name, model in models.items():  # 遍历参与解释的三个模型。
            with torch.no_grad():  # 关闭梯度，避免无意义的计算图构建。
                logits = model(inputs.to(device))  # 对当前批次执行前向推理。
                batch_predictions[model_name] = logits.argmax(dim=1).cpu()  # 保存当前模型在该批次上的预测标签。

        batch_size = inputs.size(0)  # 读取当前批次样本数。
        for index in range(batch_size):  # 逐个检查当前批次中的样本。
            if all(int(batch_predictions[name][index]) == int(targets[index]) for name in models.keys()):  # 判断该样本是否被三个模型全部预测正确。
                return inputs[index : index + 1].cpu(), int(targets[index].item())  # 返回该样本和它的真实标签编号。

    first_inputs, first_targets = next(iter(dataloader))  # 如果没找到共同预测正确的样本，则回退到测试集第一个样本。
    return first_inputs[0:1].cpu(), int(first_targets[0].item())  # 返回回退样本及真实标签编号。


def capture_feature_maps(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int, float]:  # 获取最后卷积层特征图以及对应预测结果。
    target_layer = find_last_conv_layer(model)  # 定位当前模型中最后一个卷积层。
    activations: dict[str, torch.Tensor] = {}  # 创建字典缓存前向传播产生的特征图。

    def forward_hook(_module, _inputs, output):  # 定义前向钩子函数。
        activations["value"] = output.detach()  # 将卷积层输出从计算图中分离并保存。

    handle = target_layer.register_forward_hook(forward_hook)  # 在目标卷积层上注册前向钩子。
    try:  # 使用 try/finally 确保钩子一定被移除。
        predicted_class, confidence = get_prediction(model, input_tensor, device)  # 对当前输入样本执行预测并读取置信度。
        feature_maps = activations["value"][0].cpu()  # 取出该样本的卷积特征图并移动到 CPU。
        return feature_maps, predicted_class, confidence  # 返回特征图、预测类别和置信度。
    finally:  # 无论是否成功都执行清理。
        handle.remove()  # 移除前向钩子，避免影响后续推理。


def generate_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int, float]:  # 为单个输入样本生成 Grad-CAM 热力图。
    target_layer = find_last_conv_layer(model)  # 定位用于 Grad-CAM 的目标卷积层。
    activations: dict[str, torch.Tensor] = {}  # 创建字典缓存目标层特征图。
    gradients: dict[str, torch.Tensor] = {}  # 创建字典缓存目标层梯度。

    def forward_hook(_module, _inputs, output):  # 定义前向钩子以保存卷积特征图。
        activations["value"] = output.detach()  # 断开计算图后缓存当前层输出。

    def backward_hook(_module, _grad_input, grad_output):  # 定义反向钩子以保存卷积层梯度。
        gradients["value"] = grad_output[0].detach()  # 取输出梯度并断开计算图后缓存。

    handle_forward = target_layer.register_forward_hook(forward_hook)  # 注册前向钩子。
    handle_backward = target_layer.register_full_backward_hook(backward_hook)  # 注册完整反向钩子。

    try:  # 执行前向与反向传播，随后统一清理钩子。
        model.zero_grad(set_to_none=True)  # 先清空模型中的旧梯度。
        logits = model(input_tensor.to(device))  # 对目标样本执行前向推理。
        probabilities = torch.softmax(logits, dim=1)  # 将 logits 转换为类别概率。
        predicted_class = int(probabilities.argmax(dim=1).item())  # 取最大概率类别作为预测类别。
        confidence = float(probabilities[0, predicted_class].item())  # 读取该类别概率作为置信度。
        logits[:, predicted_class].sum().backward()  # 对预测类别分数执行反向传播以生成梯度。

        activation = activations["value"][0]  # 取出该样本的卷积特征图。
        gradient = gradients["value"][0]  # 取出该样本的卷积层梯度。
        channel_weights = gradient.mean(dim=(1, 2), keepdim=True)  # 对空间维平均得到每个通道的重要性权重。
        cam = torch.relu((channel_weights * activation).sum(dim=0))  # 按通道加权求和并通过 ReLU 得到激活图。
        cam = cam.unsqueeze(0).unsqueeze(0)  # 扩展维度以便使用双线性插值。
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)  # 将 CAM 放大到原图尺寸。
        cam = cam.squeeze()  # 移除多余维度恢复成二维热力图。
        cam = cam - cam.min()  # 将最小值平移到 0。
        cam = cam / (cam.max() + 1e-8)  # 将热力图归一化到 0 到 1。
        return cam.cpu(), predicted_class, confidence  # 返回热力图、预测类别和置信度。
    finally:  # 无论前面是否出错，都执行清理逻辑。
        handle_forward.remove()  # 移除前向钩子。
        handle_backward.remove()  # 移除反向钩子。
        model.zero_grad(set_to_none=True)  # 再次清空梯度，保持模型状态干净。


def extract_vit_attention(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int, float]:  # 提取 ViT 最后一层 CLS token 到 patch 的注意力热力图。
    model.eval()  # 确保 ViT 处于评估模式。
    with torch.no_grad():  # 提取注意力时不需要计算梯度。
        x = model.patch_embed(input_tensor.to(device))  # 将输入图像转换为 patch token。
        if model.cls_token is not None:  # 如果当前模型使用 CLS token 分类。
            cls_token = model.cls_token.expand(x.size(0), -1, -1)  # 按批次扩展 CLS token。
            x = torch.cat([cls_token, x], dim=1)  # 将 CLS token 拼接到 token 序列前端。

        x = x + model.pos_embed  # 为 token 序列加入位置编码。
        x = model.pos_drop(x)  # 经过位置编码后的 dropout 层。

        if len(model.encoder) > 1:  # 如果编码器层数大于 1。
            for block in model.encoder[:-1]:  # 先执行最后一层之前的所有编码块。
                x = block(x)  # 逐层更新 token 表示。

        last_block = model.encoder[-1]  # 取出最后一个 Transformer 编码块。
        residual = x  # 保存注意力子层的残差输入。
        normalized = last_block.norm1(x)  # 对输入做第一次归一化。
        attn_out, attn_weights = last_block.attn(normalized, normalized, normalized, need_weights=True, average_attn_weights=False)  # 显式计算最后一层注意力输出与多头权重。
        x = residual + last_block.drop_path1(attn_out)  # 完成注意力子层的残差连接。
        residual = x  # 保存 MLP 子层输入。
        x = last_block.norm2(x)  # 执行第二次归一化。
        x = residual + last_block.mlp(x)  # 完成 MLP 子层的残差连接。
        x = model.final_norm(x)  # 对编码器最终输出做统一归一化。

        if model.classifier == "cls":  # 如果分类头使用 CLS token。
            pooled = x[:, 0]  # 提取 CLS token 表示作为分类输入。
            patch_attention = attn_weights[0, :, 0, 1:]  # 取所有头中 CLS token 指向 patch 的注意力。
        else:  # 如果分类方式不是 CLS token。
            pooled = x.mean(dim=1)  # 使用所有 token 的均值作为分类输入。
            patch_attention = attn_weights[0].mean(dim=1)  # 使用平均注意力作为近似可视化结果。

        logits = model.head(pooled)  # 将分类特征送入 ViT 线性分类头。
        probabilities = torch.softmax(logits, dim=1)  # 将 logits 转为概率。
        predicted_class = int(probabilities.argmax(dim=1).item())  # 读取最大概率对应的类别编号。
        confidence = float(probabilities[0, predicted_class].item())  # 读取该类别对应概率作为置信度。

        attention_map = patch_attention.mean(dim=0)  # 对多头注意力在头维上取平均。
        grid_size = model.patch_embed.grid_size  # 读取 patch 网格尺寸。
        attention_map = attention_map.reshape(grid_size, grid_size)  # 将一维 patch 权重恢复为二维网格。
        attention_map = attention_map.unsqueeze(0).unsqueeze(0)  # 扩展为四维张量以便插值。
        attention_map = F.interpolate(attention_map, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)  # 将 patch 级热力图上采样到原图大小。
        attention_map = attention_map.squeeze()  # 去除多余维度。
        attention_map = attention_map - attention_map.min()  # 将最小值平移到 0。
        attention_map = attention_map / (attention_map.max() + 1e-8)  # 将热力图归一化到 0 到 1。
        return attention_map.cpu(), predicted_class, confidence  # 返回注意力热力图、预测类别和置信度。


def save_input_image(input_tensor: torch.Tensor, class_name: str, output_path: Path) -> None:  # 保存被选中样本的原始图像。
    image = denormalize_image(input_tensor[0].cpu()).squeeze(0).numpy()  # 反归一化图像并转换为二维数组。
    fig, axis = plt.subplots(figsize=(3, 3))  # 创建一个单子图画布。
    axis.imshow(image, cmap="gray")  # 以灰度图形式显示原始字符样本。
    axis.set_title(f"Input\nTrue: {class_name}")  # 标题中写入真实类别字符。
    axis.axis("off")  # 关闭坐标轴显示。
    fig.tight_layout()  # 调整布局避免标题截断。
    fig.savefig(output_path, dpi=220, bbox_inches="tight")  # 保存图像到指定路径。
    plt.close(fig)  # 关闭图对象释放资源。


def plot_feature_maps(feature_maps: torch.Tensor, model_name: str, class_name: str, confidence: float, output_path: Path, top_k: int = 8) -> None:  # 绘制最后卷积层中响应最强的若干个通道特征图。
    channel_scores = feature_maps.mean(dim=(1, 2))  # 对每个通道在空间维求平均，作为该通道整体响应强度。
    top_indices = torch.argsort(channel_scores, descending=True)[:top_k].tolist()  # 选出响应最强的前 top_k 个通道索引。

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))  # 创建 2x4 子图布局展示前 8 个通道。
    axes = axes.flatten()  # 将二维坐标轴数组展平成一维，便于循环处理。
    for axis, channel_index in zip(axes, top_indices):  # 将每个子图与一个高响应通道配对。
        fmap = feature_maps[channel_index].numpy()  # 取出当前通道的二维特征图。
        axis.imshow(fmap, cmap="viridis")  # 使用伪彩色图显示特征响应强度。
        axis.set_title(f"ch {channel_index}")  # 标明当前显示的通道编号。
        axis.axis("off")  # 关闭坐标轴显示。

    for axis in axes[len(top_indices) :]:  # 如果展示通道数少于子图数量，则关闭剩余空子图。
        axis.axis("off")  # 关闭未使用子图。

    fig.suptitle(f"{model_name} top feature maps\nPred: {class_name} ({confidence:.3f})")  # 设置总标题，展示模型名、预测结果和置信度。
    fig.tight_layout()  # 调整布局避免标题重叠。
    fig.savefig(output_path, dpi=220, bbox_inches="tight")  # 保存特征图拼图。
    plt.close(fig)  # 关闭图对象释放资源。


def plot_gradcam(input_tensor: torch.Tensor, heatmap: torch.Tensor, model_name: str, true_class: str, predicted_class: str, confidence: float, output_path: Path) -> None:  # 绘制原图、Grad-CAM 热力图和叠加图。
    image = denormalize_image(input_tensor[0].cpu()).squeeze(0).numpy()  # 将原始输入图像反归一化并转成二维数组。
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))  # 创建三联图布局。
    axes[0].imshow(image, cmap="gray")  # 第一幅显示原始字符图像。
    axes[0].set_title(f"Input\nTrue: {true_class}")  # 标题中显示真实类别。
    axes[1].imshow(heatmap.numpy(), cmap="jet")  # 第二幅显示独立的 Grad-CAM 热力图。
    axes[1].set_title("Grad-CAM")  # 标明该图是 Grad-CAM 结果。
    axes[2].imshow(image, cmap="gray")  # 第三幅先显示原图。
    axes[2].imshow(heatmap.numpy(), cmap="jet", alpha=0.45)  # 再叠加半透明热力图展示贡献区域。
    axes[2].set_title(f"Overlay\nPred: {predicted_class} ({confidence:.3f})")  # 标题中显示预测类别和置信度。

    for axis in axes:  # 遍历三个子图坐标轴。
        axis.axis("off")  # 关闭全部坐标轴显示。

    fig.suptitle(f"{model_name} class evidence")  # 设置整张图的总标题。
    fig.tight_layout()  # 自动调整布局。
    fig.savefig(output_path, dpi=220, bbox_inches="tight")  # 保存 Grad-CAM 三联图。
    plt.close(fig)  # 关闭图对象。


def plot_vit_attention(input_tensor: torch.Tensor, attention_map: torch.Tensor, true_class: str, predicted_class: str, confidence: float, output_path: Path) -> None:  # 绘制 ViT 的原图、注意力图和叠加图。
    image = denormalize_image(input_tensor[0].cpu()).squeeze(0).numpy()  # 反归一化原始输入图像。
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))  # 创建三联图画布。
    axes[0].imshow(image, cmap="gray")  # 第一幅显示原始字符图像。
    axes[0].set_title(f"Input\nTrue: {true_class}")  # 标题中显示真实类别。
    axes[1].imshow(attention_map.numpy(), cmap="inferno")  # 第二幅显示 CLS token 到 patch 的注意力热力图。
    axes[1].set_title("CLS attention")  # 标明这是 CLS 注意力图。
    axes[2].imshow(image, cmap="gray")  # 第三幅先显示原图。
    axes[2].imshow(attention_map.numpy(), cmap="inferno", alpha=0.45)  # 再叠加注意力热力图。
    axes[2].set_title(f"Overlay\nPred: {predicted_class} ({confidence:.3f})")  # 标题中显示预测类别和置信度。

    for axis in axes:  # 遍历所有子图坐标轴。
        axis.axis("off")  # 关闭坐标轴显示。

    fig.suptitle("ViT attention map")  # 设置整张图标题。
    fig.tight_layout()  # 自动调整布局。
    fig.savefig(output_path, dpi=220, bbox_inches="tight")  # 保存注意力三联图。
    plt.close(fig)  # 关闭图对象。


def main() -> None:  # 定义主函数，负责串联模型加载、样本选择和三种可解释性可视化。
    set_seed(42)  # 固定随机种子，尽量保证输出稳定。
    device = torch.device(get_device("auto"))  # 自动选择当前可用计算设备。
    results_dir = ensure_dir(RESULTS_DIR)  # 确保输出目录存在。

    model_names = OrderedDict([("cnn", BEST_RUNS_DIR / "cnn"), ("resnet", BEST_RUNS_DIR / "resnet"), ("vit", BEST_RUNS_DIR / "vit")])  # 定义参与可解释性分析的三个模型及其目录。
    models: OrderedDict[str, torch.nn.Module] = OrderedDict()  # 初始化有序字典，用于保存已加载模型。
    configs: dict[str, dict] = {}  # 初始化字典，用于保存对应配置。
    for model_name, run_dir in model_names.items():  # 依次加载三个模型。
        model, config = load_model_from_run(run_dir, device)  # 根据配置与权重文件加载模型。
        models[model_name] = model  # 保存模型对象。
        configs[model_name] = config  # 保存模型配置。

    _, _, test_loader, _ = build_dataloaders(batch_size=configs["cnn"]["data"].get("batch_size", 128), val_ratio=configs["cnn"]["data"].get("val_ratio", 0.1), num_workers=configs["cnn"]["data"].get("num_workers", 0), seed=configs["cnn"].get("seed", 42), augmentation_config=None)  # 构建统一测试集加载器。
    class_names = list(test_loader.dataset.classes)  # 读取全部类别名称。

    input_tensor, true_label = choose_interpretability_sample(models, test_loader, device)  # 自动挑选一个三个模型都预测正确的测试样本。
    true_class = class_names[true_label]  # 根据标签编号取出真实类别字符。
    save_input_image(input_tensor, true_class, results_dir / "selected_sample.png")  # 保存被选中样本的原图。

    summary: dict[str, int | str | dict[str, float | str | int]] = {"true_label_index": true_label, "true_label": true_class}  # 初始化摘要字典，记录真实标签及各模型输出信息。

    for model_name in ("cnn", "resnet"):  # 依次处理 CNN 和 ResNet 两个卷积模型。
        feature_maps, predicted_class, confidence = capture_feature_maps(models[model_name], input_tensor, device)  # 提取最后卷积层特征图与预测结果。
        gradcam_map, gradcam_class, gradcam_confidence = generate_gradcam(models[model_name], input_tensor, device)  # 生成当前模型的 Grad-CAM 热力图。
        predicted_label = class_names[predicted_class]  # 将预测类别编号转换为字符标签。

        plot_feature_maps(feature_maps=feature_maps, model_name=model_name.upper() if model_name == "cnn" else "ResNet", class_name=predicted_label, confidence=confidence, output_path=results_dir / f"{model_name}_feature_maps.png")  # 保存响应最强的通道特征图。
        plot_gradcam(input_tensor=input_tensor, heatmap=gradcam_map, model_name=model_name.upper() if model_name == "cnn" else "ResNet", true_class=true_class, predicted_class=class_names[gradcam_class], confidence=gradcam_confidence, output_path=results_dir / f"{model_name}_gradcam.png")  # 保存当前模型的 Grad-CAM 三联图。
        summary[model_name] = {"label": class_names[gradcam_class], "confidence": gradcam_confidence, "feature_channels": int(feature_maps.shape[0]), "feature_height": int(feature_maps.shape[1]), "feature_width": int(feature_maps.shape[2])}  # 将当前模型的关键摘要写入 JSON 结果。

    vit_attention, vit_class, vit_confidence = extract_vit_attention(models["vit"], input_tensor, device)  # 提取 ViT 对当前样本的注意力热力图。
    plot_vit_attention(input_tensor=input_tensor, attention_map=vit_attention, true_class=true_class, predicted_class=class_names[vit_class], confidence=vit_confidence, output_path=results_dir / "vit_attention.png")  # 保存 ViT 注意力三联图。
    summary["vit"] = {"label": class_names[vit_class], "confidence": vit_confidence, "patch_grid": int(models["vit"].patch_embed.grid_size), "patch_size": int(models["vit"].patch_embed.patch_size)}  # 记录 ViT 的预测摘要与 patch 设置。

    save_json(summary, results_dir / "prediction_summary.json")  # 将本次可解释性分析摘要保存为 JSON。

    print(f"Selected sample true label: {true_class}")  # 在终端打印被选中样本的真实标签。
    for model_name in ("cnn", "resnet", "vit"):  # 依次打印三个模型的预测结果与置信度。
        print(f"{model_name}: pred={summary[model_name]['label']}, confidence={float(summary[model_name]['confidence']):.3f}")  # 格式化输出当前模型的预测摘要。
    print(f"Saved results to: {results_dir}")  # 在终端提示结果保存目录。


if __name__ == "__main__":  # 只有将当前文件作为主程序执行时才进入主函数。
    main()  # 调用主函数开始完整可解释性分析流程。
