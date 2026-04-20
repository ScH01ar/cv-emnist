# MLP 调参与提升过程总结

## 1. 初始 Baseline

我们首先建立了一版可以稳定训练的 MLP 基线模型，配置为：

- `hidden_dims = [512, 256, 128]`
- `activation = relu`
- `norm = none`
- `dropout = 0.2`
- `optimizer = adam`
- `scheduler = step`
- `regularization = none`

这版 baseline 的作用主要是作为后续调参与对比的统一起点。

## 2. 第一轮探索

在 baseline 的基础上，我们分别单独探索了以下因素：

- 学习率调度方法
- 激活函数
- 优化器
- 归一化方法
- L1 / L2 正则化
- Dropout 设置

第一轮的目标不是直接找到最终最优组合，而是先判断哪些技术对模型提升最明显。实验结果表明，`BatchNorm` 带来的提升最稳定，因此后续我们将它固定下来，作为下一轮搜索的新起点。

## 3. 第二轮探索

第二轮以 `BatchNorm` 配置为基础，继续逐项测试其他训练策略。  
这一轮的主要结论是：在加入 `BatchNorm` 之后，学习率调度器的影响更加明显，其中 `plateau` 的表现优于原来的 `step`，也略优于 `cosine`。

因此，第二轮结束后，我们将新的主干配置更新为：

- `relu + batchnorm + adam + plateau + dropout(0.2)`

## 4. 第三轮探索

第三轮继续在上述配置基础上，分别测试：

- 不同正则化方式
- 不同激活函数
- 不同优化器
- 不同 Dropout 设置

这一轮的结果表明，加入轻量的 `L1` 正则化后，模型性能进一步提升；而继续更换激活函数、优化器或增大 Dropout，并没有带来更好的结果。

## 5. 最终结果

最终，MLP 的最优配置为：

- `hidden_dims = [512, 256, 128]`
- `activation = relu`
- `norm = batchnorm`
- `dropout = 0.2`
- `optimizer = adam`
- `scheduler = plateau`
- `regularization = l1`
- `l1_lambda = 1e-5`

对应结果为：

- baseline `test_accuracy = 86.52%`
- `best_val_accuracy = 88.26%`
- `test_accuracy = 87.63%`

相比最初的 baseline，最终模型在测试集上的准确率提升了 `1.11` 个百分点。

## 6. 小结

整体来看，我们采用的是“从 baseline 出发，逐轮固定最优配置、再继续探索剩余因素”的调参策略。  
相比直接将所有技术自由组合，这种方法更容易分析每一种技术的实际作用，也更适合逐步找到稳定有效的配置。
