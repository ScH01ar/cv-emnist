# ViT 调参与优化路径总结

## 1. 初始 Baseline

我们首先建立了一版轻量级 Vision Transformer 作为 baseline。该 baseline 的目标不是直接取得最优结果，而是提供一个可以稳定训练、便于后续逐项调参比较的统一起点。

除特别说明外，本文所有调参实验均统一设置：

- `epochs = 30`
- `batch_size = 128`
- `val_ratio = 0.1`
- `seed = 42`

初始 baseline 配置为：

- `activation = gelu`
- `norm = layernorm`
- `dropout = 0.1`
- `optimizer = adamw`
- `scheduler = step`
- `regularization = none`
- `patch_size = 4`
- `embed_dim = 128`
- `depth = 6`
- `num_heads = 4`
- `epochs = 30`

baseline 的实验结果为：

| config | epochs | best_epoch | best_val_acc | test_acc |
|---|---:|---:|---:|---:|
| `baseline` | 30 | 16 | 0.8711 | 0.8616 |

该结果说明初始 ViT 模型已经能够正常收敛，并且可以作为后续调参的有效参照。

## 2. 第一轮：激活函数选择

第一轮实验固定其他配置不变，只改变激活函数，比较了以下几种设置：

| config | epochs | best_epoch | best_val_acc | test_acc |
|---|---:|---:|---:|---:|
| `stage1_gelu (baseline)` | 30 | 16 | 0.8711 | 0.8616 |
| `stage1_elu` | 30 | 27 | 0.8517 | 0.8438 |
| `stage1_relu` | 30 | 21 | 0.8709 | 0.8654 |
| `stage1_leakyRelu` | 30 | 30 | 0.8685 | 0.8634 |

从验证集准确率来看，`gelu` 的表现最高，虽然 `relu` 的测试集准确率略高，但超参数选择应主要依据验证集表现，而不是测试集。因此，后续实验固定使用 `gelu` 作为激活函数。

本轮结论：

- 固定 `activation = gelu`

## 3. 第二轮：优化器选择

第二轮在固定 `gelu` 的基础上，比较了不同优化器的表现。实验包括：

| config | epochs | best_epoch | best_val_acc | test_acc |
|---|---:|---:|---:|---:|
| `stage2_adamw (baseline)` | 30 | 16 | 0.8711 | 0.8616 |
| `stage2_gelu_rmsprop` | 30 | 29 | 0.7970 | 0.7923 |
| `stage2_gelu_sgd` | 30 | 28 | 0.8673 | 0.8636 |

实验结果显示，`rmsprop` 在当前 ViT 结构上表现明显较差；`sgd` 虽然能够正常训练，但验证集准确率仍低于 `adamw`。相比之下，`adamw` 在收敛速度和验证集表现上更加稳定。

因此，后续实验固定使用 `adamw` 作为优化器。

本轮结论：

- 固定 `activation = gelu`
- 固定 `optimizer = adamw`

## 4. 第三轮：正则化方式选择

第三轮在固定 `gelu + adamw` 的基础上，比较了不同正则化方式，包括不使用正则化、L1 正则化和 L2 正则化。

| config | epochs | best_epoch | best_val_acc | test_acc |
|---|---:|---:|---:|---:|
| `stage3_gelu_adamw_none (baseline)` | 30 | 16 | 0.8711 | 0.8616 |
| `stage3_gelu_adamw_l1` | 30 | 25 | 0.8677 | 0.8651 |
| `stage3_gelu_adamw_l2` | 30 | 28 | 0.8710 | 0.8634 |

结果表明，L1 和 L2 正则化都没有显著提升验证集准确率。其中 L2 的结果接近 baseline，但仍略低于不使用正则化的设置。考虑到 ViT 当前模型规模较小，并且后续还会继续调节 dropout、scheduler 和 normalization，因此本轮选择保持 `regularization = none`。

本轮结论：

- 固定 `activation = gelu`
- 固定 `optimizer = adamw`
- 固定 `regularization = none`

## 5. 第四轮：Dropout 设置选择

第四轮在固定 `gelu + adamw + no regularization` 的基础上，比较了不同 dropout 设置。

| config | epochs | best_epoch | best_val_acc | test_acc |
|---|---:|---:|---:|---:|
| `stage4_gelu_adamw_none_01 (baseline)` | 30 | 16 | 0.8711 | 0.8616 |
| `stage4_gelu_adamw_none_03` | 30 | 26 | 0.8580 | 0.8539 |
| `stage4_gelu_adamw_none_off` | 30 | 12 | 0.8647 | 0.8601 |

结果显示，较大的 dropout rate 会削弱模型拟合能力，`dropout = 0.3` 的验证集和测试集表现都明显下降；完全关闭 dropout 后，模型也没有超过 baseline。相比之下，`dropout = 0.1` 在稳定性和泛化性能之间取得了更好的平衡。

本轮结论：

- 固定 `activation = gelu`
- 固定 `optimizer = adamw`
- 固定 `regularization = none`
- 固定 `dropout = 0.1`

## 6. 第五轮：学习率调度器选择

第五轮在固定前面最优配置的基础上，比较了不同 learning rate scheduler。

| config | epochs | best_epoch | best_val_acc | test_acc |
|---|---:|---:|---:|---:|
| `stage5_gelu_adamw_none_01_step (baseline)` | 30 | 16 | 0.8711 | 0.8616 |
| `stage5_gelu_adamw_none_01_cosine` | 30 | 30 | 0.8720 | 0.8638 |
| `stage5_gelu_adamw_none_01_plateau` | 30 | 29 | 0.8672 | 0.8606 |

实验结果显示，`cosine` scheduler 的验证集准确率和测试集准确率均略高于原始 `step` scheduler，并且在训练后期仍能保持较稳定的提升趋势。`plateau` 的结果低于 `cosine`，因此后续实验选择 `cosine` scheduler。

本轮结论：

- 固定 `activation = gelu`
- 固定 `optimizer = adamw`
- 固定 `regularization = none`
- 固定 `dropout = 0.1`
- 固定 `scheduler = cosine`

## 7. 第六轮：归一化方式选择

第六轮在固定 `gelu + adamw + no regularization + dropout 0.1 + cosine` 的基础上，比较了不同 normalization 设置。

| config | epochs | best_epoch | best_val_acc | test_acc |
|---|---:|---:|---:|---:|
| `stage6_gelu_adamw_none_01_cosine_layernorm` | 30 | 30 | 0.8720 | 0.8638 |
| `stage6_gelu_adamw_none_01_cosine_batchnorm` | 30 | 27 | 0.8901 | 0.8826 |
| `stage6_gelu_adamw_none_01_cosine_none` | 30 | 30 | 0.8994 | 0.8930 |

实验结果表明，在当前 ViT 设置下，去掉 normalization 反而取得了更好的性能。`norm = none` 的验证集准确率达到 `0.8994`，测试集准确率达到 `0.8930`，明显优于前面所有配置。因此后续选择 `norm = none`。

这一结果说明，在本实验的轻量级 ViT 和 EMNIST Balanced 数据集上，LayerNorm 或 BatchNorm 并不一定带来更好的泛化效果。对于较小输入图像和较浅模型结构，去掉 normalization 可能减少了不必要的归一化约束，使模型更容易拟合字符图像中的局部形状差异。

本轮结论：

- 固定 `activation = gelu`
- 固定 `optimizer = adamw`
- 固定 `regularization = none`
- 固定 `dropout = 0.1`
- 固定 `scheduler = cosine`
- 固定 `norm = none`

## 8. ViT 结构参数补充探索

除了上述逐轮调参，我们也单独测试了 ViT 的部分结构参数，包括 patch size、模型深度和模型宽度。

| config | epochs | best_epoch | best_val_acc | test_acc |
|---|---:|---:|---:|---:|
| `baseline_patch7` | 30 | 30 | 0.8891 | 0.8804 |
| `baseline_deeper` | 30 | 30 | 0.8693 | 0.8638 |
| `baseline_wider` | 30 | 22 | 0.8745 | 0.8651 |

结构实验显示，单独增大深度并没有带来明显提升；增大模型宽度有一定改善，但提升有限。相比之下，将 `patch_size` 从 4 改为 7 的效果更明显，但其结果仍低于第六轮的 `stage6_gelu_adamw_none_01_cosine_none`。

因此，最终结构参数仍保留：

- `patch_size = 4`
- `embed_dim = 128`
- `depth = 6`
- `num_heads = 4`

## 9. 最终最佳结果

综合所有阶段实验后，最终选择的 ViT 最佳配置为：

- `activation = gelu`
- `optimizer = adamw`
- `learning_rate = 0.001`
- `weight_decay = 0.0001`
- `regularization = none`
- `dropout = 0.1`
- `scheduler = cosine`
- `min_lr = 0.00001`
- `t_max = 30`
- `norm = none`
- `patch_size = 4`
- `embed_dim = 128`
- `depth = 6`
- `num_heads = 4`
- `mlp_ratio = 4.0`
- `classifier = cls`
- `epochs = 30`

对应结果为：

- baseline `test_acc = 0.8616`
- final best `best_val_acc = 0.8994`
- final best `test_acc = 0.8930`

相比最初 baseline，最终最佳 ViT 模型在测试集准确率上提升了：

```text
0.8930 - 0.8616 = 0.0314
```

即提升约 `3.14` 个百分点。

## 10. 小结

整体来看，我们采用的是“从 baseline 出发，逐轮固定当前最优配置，再继续探索剩余因素”的调参策略。相比一次性随机组合所有技术，这种方法更容易分析每一种技术对模型性能的影响，也更适合解释模型性能提升的来源。

从实验结果看，ViT 的性能提升主要来自以下几个方面：

1. `adamw` 比 `sgd` 和 `rmsprop` 更适合当前 ViT；
2. `dropout = 0.1` 比过强 dropout 或不使用 dropout 更稳定；
3. `cosine` scheduler 比 `step` 和 `plateau` 表现更好；
4. 在当前轻量 ViT 设置下，`norm = none` 明显优于 `batchnorm` 和 `layernorm`；
5. 在结构参数中，`patch_size = 7` 单独测试时表现较强，但仍未超过第六轮最终最佳配置，因此最终保留 `patch_size = 4`。

最终 ViT 配置将测试集准确率从 baseline 的 `86.16%` 提升到 `89.30%`，说明逐轮调参策略能够有效改善模型性能。
