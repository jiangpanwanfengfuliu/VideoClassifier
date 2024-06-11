# 文件结构

- 目录：
  - 1-input_videos：原始视频文件
  - 2-covers：视频帧大小序列的csv文件
  - 3-datasets：视频帧大小序列数据集
  - 4-graphs：UMAP降维可视化分析的数据分布图
  - 5-weights：各个模型的权重文件
  - Models：BERT与VideoClassifier模型
- 文件：
  - VideoClassifier-1.py：训练数据集3-datasets/train-data
  - VideoClassifier-2.py：训练数据集3-datasets/train-data-v2，并尝试”直接使用attention层“、”提取特殊帧“、”序列向前作差“等数据处理技巧。
  - Embedding-1.py：提取3-datasets/train-data数据集经过VideoClassifier模型Embedding后的特征图。
  - Embedding-2.py：提取3-datasets/train-data-v2数据集经过VideoClassifier模型Embedding后的特征图。
  - LLM.py：尝试使用huggingface模型库中的大模型（以BERT作为尝试模型）训练数据集。

# 训练效果

- 模型训练：
  - 数据集3-datasets/train-data：![model-1](4-graphs\data-1.png)
  - 数据集3-datasets/train-data-v2：![model-1](4-graphs\data-2.png)

- UMAP降维：
  - 数据集3-datasets/train-data的原始数据：![1.v1-raw](4-graphs\1.v1-raw.png)
  - 数据集3-datasets/train-data的Embedding数据：![2.v1-embedding](4-graphs\2.v1-embedding.png)
  - 数据集3-datasets/train-data-v2原始数据：![3.v2-raw](4-graphs\3.v2-raw.png)
  - 数据集3-datasets/train-data-v2的Embedding数据：![4.v2-embedding](4-graphs\4.v2-embedding.png)

# 相关问题

- 数据集处理相关问题：
  - 提取特殊帧：这会使数据集特征丢失，减少数据集的信息。
  - 序列按时间步作差：本质上相当于右乘一个初等矩阵，而线性变换不会改变数据集的分布。
  - 可以尝试对原数据集做模式挖掘，提取一个周期的所有特征。可能可以保留原数据集的信息量，也可以使模型的输入长度不再随着视频长度的增长而增长，还可以减小模型参数的负担。
- 大模型的相关问题：
  - 不能使用预训练的参数集：如果使用大模型，这会对数据集的大小和机器的性能提出一定的要求和挑战。
  - 确定token编码规则：视频帧大小序列需要找到一种合理的方式编码为大模型能够处理的token，然后才能代入模型进行训练。
- 实验的相关问题：如果师兄希望将基于transformers的模型作为本文中ResNet相互对照的baseline，基于transformers的模型的accuracy在85%左右或者更低，即可达到论文预期的展示效果；如果师兄希望将transformers作为在下一篇文章中ResNet的替代模型，如果有幸后续通过夏令营并加入贵实验室，我将在后续的科研工作中尽全力协助师兄，在合适的设备上，训练一个更大更深更好的模型。

> 非常感谢师兄在实习期间的细心耐心指导！不胜感激！

