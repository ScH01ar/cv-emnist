# CNN / ResNet 调参与优化路径总结

## 1. CNN 优化路径

### 1.1 初始 Baseline

CNN 的初始配置使用：

- `activation = relu`
- `norm = batchnorm`
- `dropout = 0.2`
- `optimizer = adam`
- `scheduler = none`
- `regularization = l2 (1e-4)`

这一版作为后续 CNN 调参的统一对照组。

### 1.2 第一轮：优化器与学习率调度探索

第一轮主要测试训练策略本身，包括：

- 优化器：`adamw / rmsprop / sgd`（对比 baseline 的 `adam`）
- 调度器：`none / step / cosine / plateau`

从配置演进上看，`adam + step` 成为后续主干组合（并保留 `weight_decay`），说明“稳定优化器 + 分段降学习率”在 CNN 上更有效。

### 1.3 第二轮：激活函数探索

在 `adam + step` 基础上，继续比较激活函数：

- `relu`
- `elu`
- `leaky_relu`
- `gelu`

后续最佳配置沿用了 `gelu`，因此第二轮结论是：在当前训练策略下，`gelu` 比其他激活更适合该 CNN 结构。

### 1.4 第三轮：正则化、Dropout 与归一化

第三轮固定 `adam + step + gelu`，重点比较泛化控制策略：

- 正则：`l2 / l1 / none`
- Dropout：`0.0 / 0.2 / 0.5`
- 归一化：`batchnorm / none`

从最终最佳配置看，`l2 + dropout(0.2) + batchnorm` 仍是更稳健的组合；将正则去掉、关闭归一化或大幅提高 Dropout 都没有进入最终最优。

### 1.5 第四轮：数据增强验证

还测试了数据增强版本（`adam_step_gelu_aug`）。数据增强版本的最优分数虽然有所提升，但最终测试结果不如未启用增强的版本。

### 1.6 CNN 最终配置与结果

最终最佳配置（见 `artifacts/best_runs/cnn/config.yaml`）：

- `activation = gelu`
- `norm = batchnorm`
- `dropout = 0.2`
- `optimizer = adam (lr=1e-3, weight_decay=0.01)`
- `scheduler = step (step_size=5, gamma=0.1)`
- `regularization = l2 (1e-4)`

对应结果：
- baseline `test_accuracy = 88.29%`
- `best_val_accuracy = 89.52%`
- `test_accuracy = 89.14%`

---

## 2. ResNet 优化路径

### 2.1 初始 Baseline

ResNet 的初始配置为：

- `activation = relu`
- `norm = batchnorm`
- `dropout = 0.2`
- `optimizer = adam`
- `scheduler = none`
- `regularization = l2 (1e-4)`

该 baseline 主要用于和后续策略做可比性对照。

### 2.2 第一轮：优化器与学习率调度探索

第一轮先测试训练框架：

- 优化器：`adamw / rmsprop / sgd`（对比 baseline 的 `adam`）
- 调度器：`none / step / cosine / plateau`

从后续配置主线可见，`adamw + cosine` 成为稳定主干，说明 ResNet 对“解耦权重衰减 + 平滑退火学习率”的收益更明显。

### 2.3 第二轮：激活函数探索

在 `adamw + cosine` 上比较：

- `relu`
- `elu`
- `leaky_relu`
- `gelu`

后续最佳路线保留了 `gelu`，表明该激活在当前 ResNet 结构下表现更优。

### 2.4 第三轮：正则化、Dropout 与归一化

第三轮固定 `adamw + cosine + gelu`，进一步对比：

- 正则：`l2 / l1 / none`
- Dropout：`0.0 / 0.2 / 0.5`
- 归一化：`batchnorm / none`

最终最佳配置采用 `l1 + dropout(0.0) + batchnorm`，说明在当前 ResNet 上更轻的随机失活、配合轻量 L1 约束更合适。

### 2.5 第四轮：数据增强验证

同样测试了增强版（`adamw_cosine_gelu_l1_0.0_aug`）。同样，数据增强版本的最优分数虽然有所提升，但最终测试结果不如未启用增强的版本。

### 2.6 ResNet 最终配置与结果

最终最佳配置（见 `artifacts/best_runs/resnet/config.yaml`）：

- `activation = gelu`
- `norm = batchnorm`
- `dropout = 0.0`
- `optimizer = adamw (lr=1e-3, weight_decay=0.01)`
- `scheduler = cosine (t_max=15, min_lr=1e-6)`
- `regularization = l1 (1e-5)`

对应结果：
- baseline `test_accuracy = 88.62%`
- `best_val_accuracy = 90.33%`
- `test_accuracy = 90.16%`

---

## 3. 整体结论

两条优化路径都遵循了“先确定训练主干，再微调模型细节”的策略：

- CNN 最终收敛到：`adam + step + gelu + l2 + dropout(0.2)`
- ResNet 最终收敛到：`adamw + cosine + gelu + l1 + dropout(0.0)`

这也反映了两类结构的差异：CNN 更依赖显式分段降学习率与中等强度正则，ResNet 更受益于 AdamW + Cosine 的平滑训练动态与更轻的 Dropout。
