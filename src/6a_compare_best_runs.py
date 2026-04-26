from __future__ import annotations  # 启用延迟类型注解，便于在类型标注中直接引用尚未完全定义的类型。

from collections import OrderedDict  # 导入有序字典，用于固定模型遍历顺序。
from pathlib import Path  # 导入 Path，便于进行跨平台路径处理。

import matplotlib.pyplot as plt  # 导入 matplotlib，用于绘制预测样本和混淆矩阵图像。
import pandas as pd  # 导入 pandas，用于整理评估结果表格。
import seaborn as sns  # 导入 seaborn，用于绘制更美观的混淆矩阵热力图。
import torch  # 导入 PyTorch，负责模型加载、推理和张量运算。
import torch.nn.functional as F  # 导入函数式接口，用于插值上采样等操作。
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support  # 导入分类评估指标函数。

from dataset import build_dataloaders  # 导入数据加载器构建函数，用于统一读取 EMNIST 测试集。
from utils import PROJECT_ROOT, ensure_dir, get_device, load_class_from_file, load_yaml, save_json, set_seed  # 导入项目中的通用工具函数。

BEST_RUNS_DIR = PROJECT_ROOT / "artifacts" / "best_runs"  # 指定四个最优模型权重与配置所在目录。
RESULTS_DIR = PROJECT_ROOT / "results" / "best_model_comparison"  # 指定统一评估结果输出目录。
MODEL_ORDER = ("mlp", "cnn", "resnet", "vit")  # 固定四个模型的加载与展示顺序。
DISPLAY_NAMES = OrderedDict([("mlp", "MLP"), ("cnn", "CNN"), ("resnet", "ResNet"), ("vit", "ViT")])  # 定义内部目录名到展示名的映射关系。
NORMALIZE_MEAN = 0.1307  # 记录 EMNIST 归一化时使用的均值，用于可视化时反归一化。
NORMALIZE_STD = 0.3081  # 记录 EMNIST 归一化时使用的标准差，用于可视化时反归一化。


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:  # 定义反归一化函数，将标准化后的图像恢复到可显示范围。
    return image_tensor * NORMALIZE_STD + NORMALIZE_MEAN  # 按标准化公式的逆变换恢复像素值。


def load_model_from_run(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:  # 根据 best_runs 中的配置和权重加载单个模型。
    config = load_yaml(run_dir / "config.yaml")  # 读取该模型对应的配置文件。
    model_config = config["model"]  # 取出配置中的模型定义部分。
    model_class = load_class_from_file(PROJECT_ROOT / model_config["file"], model_config["class_name"])  # 通过文件路径和类名动态导入模型类。
    model = model_class(**model_config.get("kwargs", {}))  # 使用配置中的参数实例化模型。
    state_dict = torch.load(run_dir / "best.pt", map_location=device)  # 将最优权重加载到指定设备。
    model.load_state_dict(state_dict)  # 把权重写入模型实例。
    model.to(device)  # 将模型移动到 CPU 或 GPU。
    model.eval()  # 切换到评估模式，关闭 dropout 等训练行为。
    return model, config  # 返回模型对象及其配置内容。


def parameter_count(model: torch.nn.Module) -> int:  # 统计模型总参数量，便于横向比较规模差异。
    return sum(parameter.numel() for parameter in model.parameters())  # 对全部参数张量的元素个数求和。


@torch.no_grad()  # 在整个推理函数中关闭梯度计算，以减少显存和时间开销。
def predict_model(model: torch.nn.Module, dataloader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:  # 对整个测试集执行预测并返回预测标签与真实标签。
    predictions: list[torch.Tensor] = []  # 初始化用于保存每个批次预测结果的列表。
    targets: list[torch.Tensor] = []  # 初始化用于保存每个批次真实标签的列表。

    for inputs, labels in dataloader:  # 遍历测试集中的每一个批次。
        logits = model(inputs.to(device))  # 将输入移动到设备后送入模型得到输出 logits。
        predictions.append(logits.argmax(dim=1).cpu())  # 取每个样本概率最大的类别作为预测并移回 CPU。
        targets.append(labels.cpu())  # 将当前批次真实标签保存在 CPU 侧列表中。

    return torch.cat(predictions), torch.cat(targets)  # 将所有批次沿样本维拼接成完整测试集结果。


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:  # 计算 accuracy、precision、recall 和 F1 指标。
    pred_np = predictions.numpy()  # 将预测张量转成 numpy，便于传入 sklearn。
    target_np = targets.numpy()  # 将真实标签张量转成 numpy。
    precision, recall, f1, _ = precision_recall_fscore_support(target_np, pred_np, average="macro", zero_division=0)  # 计算宏平均精确率、召回率和 F1。
    return {"accuracy": accuracy_score(target_np, pred_np), "precision": precision, "recall": recall, "f1": f1}  # 以字典形式返回四个指标。


def build_sample_prediction_table(sample_targets: torch.Tensor, sample_predictions: dict[str, torch.Tensor], class_names: list[str]) -> pd.DataFrame:  # 将前 6 个样本的真实标签和四个模型预测整理成表格。
    rows = []  # 初始化用于保存每一行样本记录的列表。
    for index in range(len(sample_targets)):  # 遍历前 6 个样本中的每一个位置。
        row = {"sample_index": index, "true_label": class_names[int(sample_targets[index])]}  # 先写入样本编号和真实标签。
        for model_key, predictions in sample_predictions.items():  # 继续写入每个模型在该样本上的预测标签。
            row[DISPLAY_NAMES[model_key]] = class_names[int(predictions[index])]  # 将内部键名映射为展示名并记录标签字符。
        rows.append(row)  # 将该样本对应的一行结果加入列表。
    return pd.DataFrame(rows)  # 把行列表转换成 DataFrame 便于打印和保存。


def plot_top6_predictions(sample_inputs: torch.Tensor, sample_targets: torch.Tensor, sample_predictions: dict[str, torch.Tensor], class_names: list[str], output_path: Path) -> None:  # 将前 6 个样本及四个模型的预测结果画成对比图。
    row_count = 1 + len(sample_predictions)  # 第一行用于显示真实标签，后续每行对应一个模型。
    fig, axes = plt.subplots(row_count, 6, figsize=(18, 3.4 * row_count))  # 创建规则网格子图，列数固定为 6。

    for column in range(6):  # 先绘制第一行真实图像与真实标签。
        image = denormalize_image(sample_inputs[column]).squeeze(0).cpu().numpy()  # 取出图像并反归一化为可显示灰度图。
        axes[0, column].imshow(image, cmap="gray")  # 使用灰度色图显示字符图像。
        axes[0, column].set_title(f"True: {class_names[int(sample_targets[column])]}")  # 在标题中写出真实标签。
        axes[0, column].axis("off")  # 关闭坐标轴以减少干扰。

    for row_index, (model_key, predictions) in enumerate(sample_predictions.items(), start=1):  # 依次绘制各个模型的预测结果。
        for column in range(6):  # 在当前模型行中遍历 6 个样本。
            image = denormalize_image(sample_inputs[column]).squeeze(0).cpu().numpy()  # 取出当前样本图像并反归一化。
            predicted_label = class_names[int(predictions[column])]  # 获取当前模型的预测标签字符。
            true_label = class_names[int(sample_targets[column])]  # 获取对应样本的真实标签字符。
            correct = predicted_label == true_label  # 判断当前预测是否正确。
            axes[row_index, column].imshow(image, cmap="gray")  # 再次显示原图，以便直接对应查看预测结果。
            axes[row_index, column].set_title(f"{DISPLAY_NAMES[model_key]}: {predicted_label}", color="green" if correct else "red")  # 使用颜色区分预测正确与错误。
            axes[row_index, column].axis("off")  # 关闭坐标轴显示。

    fig.tight_layout()  # 自动调整子图间距，避免标题重叠。
    fig.savefig(output_path, dpi=200, bbox_inches="tight")  # 将图像保存到指定路径。
    plt.close(fig)  # 关闭当前图对象，释放内存。


def plot_confusion_matrices(predictions_by_model: dict[str, torch.Tensor], targets: torch.Tensor, class_names: list[str], output_path: Path) -> None:  # 绘制四个模型的混淆矩阵拼图。
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))  # 创建 2x2 布局用于同时展示四个模型。
    axes = axes.flatten()  # 将二维坐标轴数组展平成一维，便于循环访问。

    target_np = targets.numpy()  # 将真实标签转成 numpy 数组。
    tick_positions = list(range(0, len(class_names), 5))  # 为了避免标签太密，只每隔 5 个类显示一个刻度。
    tick_labels = [class_names[index] for index in tick_positions]  # 构造对应的刻度标签字符。

    for axis, (model_key, predictions) in zip(axes, predictions_by_model.items()):  # 将每个模型与一个子图坐标轴配对。
        cm = confusion_matrix(target_np, predictions.numpy(), labels=list(range(len(class_names))))  # 计算当前模型的完整混淆矩阵。
        sns.heatmap(cm, ax=axis, cmap="Blues", cbar=False, square=True, xticklabels=False, yticklabels=False)  # 使用 seaborn 绘制热力图。
        axis.set_title(f"{DISPLAY_NAMES[model_key]} Confusion Matrix")  # 设置当前子图标题。
        axis.set_xlabel("Predicted label")  # 标注横轴含义为预测标签。
        axis.set_ylabel("True label")  # 标注纵轴含义为真实标签。
        axis.set_xticks([position + 0.5 for position in tick_positions], tick_labels, rotation=0)  # 设置横轴主要刻度位置与标签。
        axis.set_yticks([position + 0.5 for position in tick_positions], tick_labels, rotation=0)  # 设置纵轴主要刻度位置与标签。

    fig.tight_layout()  # 调整子图布局以避免重叠。
    fig.savefig(output_path, dpi=220, bbox_inches="tight")  # 保存混淆矩阵总图。
    plt.close(fig)  # 关闭图像对象。


def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:  # 在 CNN 或 ResNet 中寻找最后一个卷积层，供 Grad-CAM 使用。
    for module in reversed(list(model.modules())):  # 倒序遍历模型中的所有子模块。
        if isinstance(module, torch.nn.Conv2d):  # 如果当前模块是二维卷积层。
            return module  # 立即返回这个最后出现的卷积层。
    raise ValueError(f"No Conv2d layer found in {model.__class__.__name__}.")  # 如果没找到卷积层则抛出异常。


def generate_gradcam_overlay(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int]:  # 计算单张输入图像的 Grad-CAM 热力图及其预测类别。
    target_layer = find_last_conv_layer(model)  # 先定位可用于 Grad-CAM 的目标卷积层。
    activations: dict[str, torch.Tensor] = {}  # 创建字典保存前向传播得到的特征图。
    gradients: dict[str, torch.Tensor] = {}  # 创建字典保存反向传播得到的梯度。

    def forward_hook(_module, _inputs, output):  # 定义前向钩子，用于缓存目标层输出。
        activations["value"] = output.detach()  # 将特征图从计算图中分离后保存。

    def backward_hook(_module, _grad_input, grad_output):  # 定义反向钩子，用于缓存目标层梯度。
        gradients["value"] = grad_output[0].detach()  # 取输出梯度并从计算图中分离后保存。

    handle_forward = target_layer.register_forward_hook(forward_hook)  # 注册前向钩子。
    handle_backward = target_layer.register_full_backward_hook(backward_hook)  # 注册完整反向钩子。

    try:  # 开始执行前向和反向传播，确保最后一定能移除钩子。
        model.zero_grad(set_to_none=True)  # 清空模型现有梯度，避免被历史信息污染。
        logits = model(input_tensor.to(device))  # 对单张输入图像进行前向计算。
        predicted_class = int(logits.argmax(dim=1).item())  # 取 logits 最大的类别作为预测类别。
        score = logits[:, predicted_class].sum()  # 提取目标类别分数，作为反向传播目标。
        score.backward()  # 对目标类别分数执行反向传播，得到梯度。

        activation = activations["value"][0]  # 取出当前样本在目标层上的特征图。
        gradient = gradients["value"][0]  # 取出当前样本在目标层上的梯度。
        channel_weights = gradient.mean(dim=(1, 2), keepdim=True)  # 按空间维做平均，得到每个通道的重要性权重。
        cam = torch.relu((channel_weights * activation).sum(dim=0))  # 按通道加权求和后经过 ReLU，生成类激活图。
        cam = cam.unsqueeze(0).unsqueeze(0)  # 扩展成四维张量，便于后续插值。
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)  # 上采样到原图大小。
        cam = cam.squeeze()  # 去掉多余维度，恢复到二维热力图。
        cam = cam - cam.min()  # 将热力图最小值移动到 0。
        cam = cam / (cam.max() + 1e-8)  # 将热力图归一化到 0 到 1 范围。
        return cam.cpu(), predicted_class  # 返回 CPU 上的热力图和预测类别编号。
    finally:  # 无论前面是否报错，都执行清理逻辑。
        handle_forward.remove()  # 移除前向钩子，避免影响后续推理。
        handle_backward.remove()  # 移除反向钩子，避免产生副作用。
        model.zero_grad(set_to_none=True)  # 再次清空梯度，保持模型状态干净。


def plot_gradcam(model_name: str, model: torch.nn.Module, input_tensor: torch.Tensor, true_label: int, class_names: list[str], device: torch.device, output_path: Path) -> None:  # 将单个模型的 Grad-CAM 可视化保存为三联图。
    cam, predicted_class = generate_gradcam_overlay(model, input_tensor, device)  # 先计算当前输入图像的热力图和预测类别。
    image = denormalize_image(input_tensor[0].cpu()).squeeze(0).numpy()  # 将原图反归一化并转为二维数组。

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))  # 创建三个并排子图用于展示原图、热力图和叠加图。
    axes[0].imshow(image, cmap="gray")  # 第一幅图显示原始灰度字符图像。
    axes[0].set_title(f"Input\nTrue: {class_names[true_label]}")  # 标题中展示真实标签。
    axes[1].imshow(cam.numpy(), cmap="jet")  # 第二幅图显示独立的 Grad-CAM 热力图。
    axes[1].set_title("Grad-CAM")  # 为热力图子图设置标题。
    axes[2].imshow(image, cmap="gray")  # 第三幅图先显示原图作为底图。
    axes[2].imshow(cam.numpy(), cmap="jet", alpha=0.45)  # 再叠加半透明热力图显示关注区域。
    axes[2].set_title(f"Overlay\nPred: {class_names[predicted_class]}")  # 显示预测标签。

    for axis in axes:  # 遍历三个子图坐标轴。
        axis.axis("off")  # 全部关闭坐标轴显示。

    fig.suptitle(f"{model_name} attention region")  # 设置整张图的总标题。
    fig.tight_layout()  # 调整布局，避免子图文字重叠。
    fig.savefig(output_path, dpi=220, bbox_inches="tight")  # 保存 Grad-CAM 三联图。
    plt.close(fig)  # 关闭图对象释放资源。


def get_vit_attention_map(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int]:  # 提取 ViT 最后一层中 CLS token 对各 patch 的注意力热力图。
    model.eval()  # 确保模型处于评估模式。
    with torch.no_grad():  # 提取注意力时不需要记录梯度。
        x = model.patch_embed(input_tensor.to(device))  # 先将图像分块并映射成 patch token 序列。
        if model.cls_token is not None:  # 如果当前 ViT 使用 CLS 分类头。
            cls_token = model.cls_token.expand(x.size(0), -1, -1)  # 扩展 CLS token 以匹配批次大小。
            x = torch.cat([cls_token, x], dim=1)  # 将 CLS token 拼接到 patch token 序列开头。

        x = x + model.pos_embed  # 加入位置编码，使模型区分不同 patch 的空间位置。
        x = model.pos_drop(x)  # 经过位置编码后的 dropout 层。

        if len(model.encoder) > 1:  # 如果编码器不止一层。
            for block in model.encoder[:-1]:  # 先执行除最后一层以外的全部编码块。
                x = block(x)  # 将 token 序列逐层更新。

        last_block = model.encoder[-1]  # 取出最后一个 Transformer 编码块。
        residual = x  # 保存残差连接输入。
        normalized = last_block.norm1(x)  # 先做第一阶段归一化。
        attn_out, attn_weights = last_block.attn(normalized, normalized, normalized, need_weights=True, average_attn_weights=False)  # 显式获取多头注意力输出和权重。
        x = residual + last_block.drop_path1(attn_out)  # 完成注意力子层的残差连接。
        residual = x  # 保存 MLP 子层输入作为下一次残差。
        x = last_block.norm2(x)  # 执行第二次归一化。
        x = residual + last_block.mlp(x)  # 完成 MLP 子层的残差连接。
        x = model.final_norm(x)  # 对最后输出做整体归一化。

        if model.classifier == "cls":  # 如果分类方式是 CLS token。
            logits = model.head(x[:, 0])  # 取 CLS token 表示送入分类头。
            cls_attention = attn_weights[0, :, 0, 1:]  # 提取所有头中 CLS token 指向各 patch 的注意力。
        else:  # 如果分类方式是 patch 平均池化。
            logits = model.head(x.mean(dim=1))  # 先对 token 求均值再送入分类头。
            cls_attention = attn_weights[0].mean(dim=1)  # 使用头内平均注意力作为近似可视化依据。

        predicted_class = int(logits.argmax(dim=1).item())  # 获取 ViT 对当前样本的预测类别编号。
        attention_map = cls_attention.mean(dim=0)  # 将多个头的注意力按头维求平均。
        grid_size = model.patch_embed.grid_size  # 读取 patch 网格宽高。
        attention_map = attention_map.reshape(grid_size, grid_size)  # 将一维 patch 注意力恢复成二维网格。
        attention_map = attention_map.unsqueeze(0).unsqueeze(0)  # 扩展成四维张量便于插值。
        attention_map = F.interpolate(attention_map, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)  # 上采样到原图大小。
        attention_map = attention_map.squeeze()  # 移除多余维度恢复成二维热力图。
        attention_map = attention_map - attention_map.min()  # 先将最小值平移到 0。
        attention_map = attention_map / (attention_map.max() + 1e-8)  # 将热力图归一化到 0 到 1 范围。
        return attention_map.cpu(), predicted_class  # 返回 CPU 上的注意力图和预测类别编号。


def plot_vit_attention(model: torch.nn.Module, input_tensor: torch.Tensor, true_label: int, class_names: list[str], device: torch.device, output_path: Path) -> None:  # 将 ViT 的注意力热力图保存为三联图。
    attention_map, predicted_class = get_vit_attention_map(model, input_tensor, device)  # 获取当前输入样本的注意力热力图和预测类别。
    image = denormalize_image(input_tensor[0].cpu()).squeeze(0).numpy()  # 将原图反归一化为可显示的二维数组。

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))  # 创建原图、注意力图和叠加图三个子图。
    axes[0].imshow(image, cmap="gray")  # 第一幅图显示原始输入字符图像。
    axes[0].set_title(f"Input\nTrue: {class_names[true_label]}")  # 标题显示真实标签。
    axes[1].imshow(attention_map.numpy(), cmap="inferno")  # 第二幅图显示 ViT 注意力热力图。
    axes[1].set_title("CLS attention")  # 标注这是 CLS token 的注意力。
    axes[2].imshow(image, cmap="gray")  # 第三幅图先显示原图。
    axes[2].imshow(attention_map.numpy(), cmap="inferno", alpha=0.45)  # 再叠加半透明注意力热力图。
    axes[2].set_title(f"Overlay\nPred: {class_names[predicted_class]}")  # 标题中显示预测标签。

    for axis in axes:  # 遍历全部子图。
        axis.axis("off")  # 关闭坐标轴显示。

    fig.suptitle("ViT attention region")  # 设置整张图标题。
    fig.tight_layout()  # 自动调整布局。
    fig.savefig(output_path, dpi=220, bbox_inches="tight")  # 保存三联图到结果目录。
    plt.close(fig)  # 关闭图像对象。


def main() -> None:  # 定义脚本主入口，统一执行模型加载、评估和可视化流程。
    set_seed(42)  # 固定随机种子，保证结果尽量可复现。
    device = torch.device(get_device("auto"))  # 自动选择当前可用设备，优先使用 GPU。
    results_dir = ensure_dir(RESULTS_DIR)  # 确保结果输出目录存在。

    run_configs: dict[str, dict] = {}  # 初始化字典，用于保存每个模型的配置内容。
    models: OrderedDict[str, torch.nn.Module] = OrderedDict()  # 初始化有序字典，用于按固定顺序保存模型对象。
    parameter_rows: list[dict[str, int | str]] = []  # 初始化列表，用于汇总参数量统计表。

    for model_key in MODEL_ORDER:  # 依次加载四个最优模型。
        run_dir = BEST_RUNS_DIR / model_key  # 计算当前模型 best_runs 子目录路径。
        model, config = load_model_from_run(run_dir, device)  # 根据配置和权重文件加载模型。
        models[model_key] = model  # 将模型对象保存到有序字典中。
        run_configs[model_key] = config  # 保存当前模型对应配置。
        parameter_rows.append({"model_name": DISPLAY_NAMES[model_key], "parameters": parameter_count(model)})  # 统计参数量并加入结果列表。

    _, _, test_loader, _ = build_dataloaders(batch_size=run_configs["cnn"]["data"].get("batch_size", 128), val_ratio=run_configs["cnn"]["data"].get("val_ratio", 0.1), num_workers=run_configs["cnn"]["data"].get("num_workers", 0), seed=run_configs["cnn"].get("seed", 42), augmentation_config=None)  # 使用统一配置构建测试集加载器。
    class_names = list(test_loader.dataset.classes)  # 从数据集对象中读取全部类别名称。

    first_inputs, first_targets = next(iter(test_loader))  # 取测试集中的第一个批次作为前 6 个样本展示来源。
    sample_inputs = first_inputs[:6].cpu()  # 截取前 6 个输入样本并移到 CPU。
    sample_targets = first_targets[:6].cpu()  # 截取前 6 个真实标签并移到 CPU。

    prediction_rows: dict[str, torch.Tensor] = OrderedDict()  # 初始化字典，用于保存每个模型在测试集上的全量预测。
    metrics_rows: list[dict[str, float | str]] = []  # 初始化列表，用于保存每个模型的四项指标结果。
    all_targets: torch.Tensor | None = None  # 预留变量保存统一的真实标签序列。

    for model_key, model in models.items():  # 依次评估四个模型在整个测试集上的表现。
        predictions, targets = predict_model(model, test_loader, device)  # 获取当前模型的全量预测和真实标签。
        if all_targets is None:  # 只在第一次循环时写入真实标签。
            all_targets = targets  # 保存测试集统一真实标签。
        prediction_rows[model_key] = predictions  # 记录当前模型的全量预测结果。
        metrics = compute_metrics(predictions, targets)  # 计算当前模型的四项指标。
        metrics_rows.append({"model_name": DISPLAY_NAMES[model_key], **metrics})  # 将模型名称和四项指标一起加入结果表。

    assert all_targets is not None  # 确保真实标签已经被成功初始化。

    sample_prediction_rows = OrderedDict()  # 初始化有序字典，用于保存前 6 个样本的逐模型预测。
    with torch.no_grad():  # 在前 6 个样本推理时关闭梯度。
        for model_key, model in models.items():  # 遍历四个模型。
            logits = model(sample_inputs.to(device))  # 对前 6 个样本执行前向推理。
            sample_prediction_rows[model_key] = logits.argmax(dim=1).cpu()  # 保存每个模型对前 6 个样本的预测标签。

    metrics_df = pd.DataFrame(metrics_rows).sort_values("accuracy", ascending=False).reset_index(drop=True)  # 构建指标总表并按准确率降序排列。
    parameter_table = pd.DataFrame(parameter_rows)  # 构建参数量统计表。
    sample_table = build_sample_prediction_table(sample_targets, sample_prediction_rows, class_names)  # 构建前 6 个样本预测对比表。

    print("\nTop-6 sample predictions")  # 在终端打印标题，提示接下来展示样本预测。
    print(sample_table.to_string(index=False))  # 以整齐表格形式打印前 6 个样本预测结果。
    print("\nOverall metrics")  # 在终端打印总指标标题。
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))  # 以四位小数格式打印模型指标表。

    metrics_df.to_csv(results_dir / "metrics_summary.csv", index=False)  # 保存四个模型指标总表为 CSV。
    parameter_table.to_csv(results_dir / "parameter_summary.csv", index=False)  # 保存参数量统计表为 CSV。
    sample_table.to_csv(results_dir / "top6_predictions.csv", index=False)  # 保存前 6 个样本预测对比表为 CSV。
    save_json(metrics_df.to_dict(orient="records"), results_dir / "metrics_summary.json")  # 额外保存一份 JSON 格式指标结果，便于其他脚本读取。

    plot_top6_predictions(sample_inputs=sample_inputs, sample_targets=sample_targets, sample_predictions=sample_prediction_rows, class_names=class_names, output_path=results_dir / "top6_predictions.png")  # 生成并保存前 6 个样本预测对比图。
    plot_confusion_matrices(predictions_by_model=prediction_rows, targets=all_targets, class_names=class_names, output_path=results_dir / "confusion_matrices.png")  # 生成并保存四模型混淆矩阵拼图。

    focus_input = sample_inputs[0:1]  # 取第一个样本作为可解释性可视化的统一输入。
    focus_label = int(sample_targets[0].item())  # 读取该样本的真实标签编号。
    plot_gradcam(model_name="CNN", model=models["cnn"], input_tensor=focus_input, true_label=focus_label, class_names=class_names, device=device, output_path=results_dir / "cnn_gradcam.png")  # 为 CNN 生成并保存 Grad-CAM 图像。
    plot_gradcam(model_name="ResNet", model=models["resnet"], input_tensor=focus_input, true_label=focus_label, class_names=class_names, device=device, output_path=results_dir / "resnet_gradcam.png")  # 为 ResNet 生成并保存 Grad-CAM 图像。
    plot_vit_attention(model=models["vit"], input_tensor=focus_input, true_label=focus_label, class_names=class_names, device=device, output_path=results_dir / "vit_attention.png")  # 为 ViT 生成并保存注意力热力图。

    print(f"\nSaved results to: {results_dir}")  # 在终端输出结果目录位置，便于快速查看文件。


if __name__ == "__main__":  # 仅当脚本作为主程序执行时才运行主函数。
    main()  # 调用主函数，开始完整评估流程。
