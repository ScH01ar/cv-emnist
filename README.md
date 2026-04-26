# cv-emnist

EMNIST Balanced 字符图像分类项目。仓库包含四类模型的训练配置、公共训练框架、最佳模型结果、公共实验结果和最终 notebook。

## 代码说明

- `models/`：模型结构，包括 `MLP / CNN / ResNet / ViT`
- `configs/`：各模型训练配置和小样本实验配置
- `src/`：公共代码，包括数据加载、训练入口、评估和公共实验脚本
- `runs/`：本地训练输出目录，默认不作为最终结果提交
- `artifacts/best_runs/`：四个模型的最佳权重、配置和训练日志
- `artifacts/6.a/`：四个模型统一测试评估结果
- `artifacts/6.b/`：可解释性分析结果
- `artifacts/6_c/`：鲁棒性评估结果
- `docs/`：各模型调参总结和公共实验分析文档
- `Group_21.ipynb`：最终整理用 notebook

## 操作说明

开始开发前建议从最新 `main` 拉分支：

```bash
git checkout main
git pull
git checkout -b your-branch-name
```

训练单个配置：

```bash
python src/main.py --config configs/mlp/baseline.yaml
```

训练结果会保存在：

```text
runs/<run_name>/
```

每个 run 默认包含：

- `best.pt`
- `history.json`
- `summary.json`
- `test_metrics.json`

小样本实验可以直接运行对应配置，例如：

```bash
python src/main.py --config configs/mlp/small_30.yaml
python src/main.py --config configs/cnn/small_30.yaml
python src/main.py --config configs/resnet/small_30.yaml
```

ViT 小样本实验使用批量入口：

```bash
python src/run_small_data.py --config configs/vit/small_base.yaml --ratios 0.3 0.5 1.0
```

## 最佳结果

当前四个模型的最佳结果保存在 `artifacts/best_runs/<model>/`，每个目录包含：

- `best.pt`
- `config.yaml`
- `history.json`
- `summary.json`
- `test_metrics.json`

当前 best run 测试集准确率：

| Model | Test Accuracy |
|---|---:|
| ResNet | 90.16% |
| ViT | 89.30% |
| CNN | 89.14% |
| MLP | 88.65% |

如果后续重新训练出更好的模型，只提交对应模型的 `artifacts/best_runs/<model>/`，不要提交完整 `runs/`。

## 公共实验

公共实验已经整理到脚本、`artifacts` 和 notebook 中：

- `5.c`：最佳模型训练曲线、训练时间和内存占用，展示在 `Group_21.ipynb`
- `5.d`：小样本训练分析，展示在 `Group_21.ipynb`
- `6.a`：统一测试集评估与模型比较，结果在 `artifacts/6.a/`
- `6.b`：可解释性分析，结果在 `artifacts/6.b/`
- `6.c`：鲁棒性评估，结果在 `artifacts/6_c/`

公共实验脚本：

```bash
python src/6a_compare_best_runs.py
python src/6b_interpretability_analysis.py
python src/6c_robustness_evaluate.py --batch-size 512 --num-workers 0 --device auto
```

最终提交和展示主要看：

- `Group_21.ipynb`
- `artifacts/best_runs/`
- `artifacts/6.a/`
- `artifacts/6.b/`
- `artifacts/6_c/`
