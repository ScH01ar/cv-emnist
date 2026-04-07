# cv-emnist

EMNIST Balanced 项目骨架。

## 代码说明

- `models/`：模型结构
- `configs/`：各模型的训练配置
- `src/`：公共代码
- `runs/`：训练输出

当前公共部分：

- `src/dataset.py`：数据读取和 dataloader
- `src/trainer.py`：训练和测试接口
- `src/utils.py`：通用工具函数
- `src/main.py`：统一训练入口


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
3. 用统一入口训练：

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

## TODO

- 目前只完成了模型训练阶段的公共代码
- 补模型间对比实验相关模块
- 补更完整的评测与结果整理流程
