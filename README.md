# cv-emnist

EMNIST Balanced 项目骨架。

## 代码说明

- `models/`：模型结构
- `configs/`：各模型的训练配置
- `src/`：公共代码
- `runs/`：训练输出
- `artifacts/`：最终保留的最佳实验结果

当前公共部分：

- `src/dataset.py`：数据读取和 dataloader
- `src/trainer.py`：训练策略、优化器、学习率调度、正则化
- `src/utils.py`：通用工具函数
- `src/main.py`：统一训练入口
- `src/run_small_data.py`：小样本实验批量入口

当前已支持的训练策略：

1. 自适应学习率（Learning Rate Schedulers）
   - `none`
   - `step`
   - `cosine`
   - `plateau`
   - `exponential`
2. 激活函数（Activation Function）
   - `relu`
   - `leaky_relu`
   - `elu`
   - `gelu`
   - `silu`
3. 优化器（Optimizers）
   - `sgd`
   - `adam`
   - `adamw`
   - `rmsprop`
   - `asgd`
   - `adagrad`
4. 归一化（Normalization）
   - `none`
   - `batchnorm`
   - `layernorm`
5. L1 与 L2 正则化（Regularization）
   - `none`
   - `l1`
   - `l2`
6. Dropout
   - 在 `model.kwargs` 中配置 `dropout`
   - `0.0` 表示不使用 Dropout
   - 其他数值表示使用对应 Dropout rate

其中 `optimizer / scheduler / regularization` 由公共 `trainer` 支持，`activation / normalization / dropout` 需要对应模型文件实现并在配置中传入。


不要把数据读取、训练循环、结果保存这些公共逻辑重复写进各自模型文件里。模型相关修改尽量只放在 `models/` 和对应 `configs/` 中。

## 操作说明

开始自己的开发前，先从 `main` 拉一个新分支：

```bash
git checkout main
git pull
git checkout -b your-branch-name
```

成员日常开发流程：

1. 在自己负责的 `models/*.py` 中写模型。
2. 在对应的 `configs/*/` 下新建或修改配置文件。
3. 模型结构相关参数写在 `model.kwargs`。
4. 训练策略相关参数写在 `train` 里，例如 `optimizer / scheduler / regularization`。
5. 用统一入口训练：

```bash
python src/main.py --config configs/xxx/xxx.yaml
```

当前已经给了一份可运行的最简 baseline：

- `models/mlp.py`
- `configs/mlp/baseline.yaml`

训练结果保存在：

```text
runs/<run_name>/
```

默认输出：

- `best.pt`
- `history.json`
- `summary.json`
- `test_metrics.json`

开发约定：

- 不直接在 `main` 分支上开发
- 公共代码改动先沟通
- 模型代码和配置优先各自维护，避免互相覆盖

## 公共实验

当前需要统一处理的公共实验有：

1. 小样本实验
2. 鲁棒性实验
3. 可解释性分析

说明：

- 统一测试评估、混淆矩阵、预测样本展示、最终绘图，后续统一在 `.ipynb` 中整理
- 这里优先实现需要单独跑实验的公共部分

### 1. 小样本实验

状态：已实现

目的：

- 使用各模型的最优配置，分别跑 `30% / 50% / 100%` 训练数据
- 比较不同数据规模下的性能变化

基本操作：

```bash
python src/run_small_data.py --config configs/xxx/xxx_best.yaml
```

例如：

```bash
python src/run_small_data.py --config configs/mlp/stage3_regularization_l1.yaml
```

默认会生成三组实验：

- `xxx_small_30`
- `xxx_small_50`
- `xxx_small_100`

并额外输出一份汇总结果：

```text
runs/<base_run_name>_small_data_summary.json
```

### 2. 鲁棒性实验

状态：未实现

目的：

- 在测试集上加入旋转、高斯噪声、模糊等扰动
- 比较四个模型在扰动条件下的性能下降情况

基本操作：

### 3. 可解释性分析

状态：未实现

目的：

- `CNN / ResNet`：Grad-CAM 或 feature maps
- `ViT`：attention heatmap

基本操作：

## 最佳 Run 收集

每个人在确定自己模型的最优配置后，不要提交整个 `runs/`，只需要把该模型的最优 `run` 目录内容复制到：

```text
artifacts/best_runs/<model>/
```

例如：

- `artifacts/best_runs/mlp/`
- `artifacts/best_runs/cnn/`
- `artifacts/best_runs/resnet/`
- `artifacts/best_runs/vit/`

目录中至少需要包含：

- `best.pt`
- `history.json`
- `summary.json`
- `test_metrics.json`

例如把某个最优实验结果复制到 `cnn` 目录：

```bash
mkdir -p artifacts/best_runs/cnn
cp runs/cnn_best_run/best.pt artifacts/best_runs/cnn/
cp runs/cnn_best_run/history.json artifacts/best_runs/cnn/
cp runs/cnn_best_run/summary.json artifacts/best_runs/cnn/
cp runs/cnn_best_run/test_metrics.json artifacts/best_runs/cnn/
```

这样后续统一做：

- 曲线绘制
- 结果汇总
- notebook 整理
- 鲁棒性实验
- 可解释性分析

都会更方便。

## TODO

- 补鲁棒性实验公共模块
- 补可解释性分析公共模块
- 在 notebook 中统一整理测试评估、绘图和结果展示
