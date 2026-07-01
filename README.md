# Machine Learning Workbench

这是一个基于原生 C++ 实现的 MNIST 分类实验平台，提供命令行界面和 Qt 桌面界面，支持对多种经典分类算法进行统一训练、测试和可视化展示。

当前实现的模型包括：

- Support Vector Machine, `SVM`
- Fully Connected Neural Network, `FCNN / BP`
- Convolutional Neural Network, `CNN`
- Logistic Regression, `LR`
- Random Forest, `RF`
- K-Nearest Neighbors, `KNN`

项目的核心目标不是只训练某一个模型，而是构建一个统一的实验工作台，让这些模型使用同一套 MNIST 数据、同一套参数文件管理方式、同一套评估指标与可视化流程。

## 1. 项目整体结构

仓库中的主要文件和目录如下：

- `Models.cpp`
  整个项目的核心实现文件。包含：
  - MNIST 数据读取
  - 6 类模型的实现
  - 训练、预测、测试逻辑
  - 混淆矩阵、Precision、Recall、F1、AUC、ROC 数据生成
- `ModelInterface.h`
  定义统一模型接口 `IClassificationModel` 与模型工厂 `ModelFactory`
- `cli_main.cpp`
  命令行程序入口
- `build-qt/gui-src/AppBackend.h`
  Qt 前端与模型核心之间的数据结构和接口声明
- `build-qt/gui-src/AppBackend.cpp`
  Qt 前端的后端桥接层，负责加载数据集、调用模型训练/测试、读取 ROC 数据
- `build-qt/gui-src/main.cpp`
  Qt 图形界面主程序，包含训练页、测试页、结构可视化页
- `build-qt/gui-src/CMakeLists.txt`
  Qt 工程构建脚本，生成 `cnn_gui` 和 `cnn_cli`
- `Mathine Learning.cmd`
  Windows 下的桌面程序启动脚本

从架构上看，这个项目可以分成 4 层：

1. 数据层
   负责读取 MNIST 原始二进制文件并做输入归一化。
2. 模型抽象层
   用统一接口约束所有模型的训练、预测、保存、加载和评估行为。
3. 算法实现层
   在 `Models.cpp` 中实现所有具体模型。
4. 展示与交互层
   包括 CLI 和 Qt GUI，负责训练控制、测试展示和模型结构可视化。

## 2. 统一接口设计

所有模型都继承 `IClassificationModel`，在 `ModelInterface.h` 中定义了统一行为，包括：

- `getName()`
- `getDescription()`
- `getStructureDescription()`
- `initWeights()`
- `loadParams()`
- `saveParams()`
- `train()`
- `predict()`
- `predictProba()`
- `evaluate()`
- `computeConfusionMatrix()`
- `computeMetrics()`
- `computeAUC()`
- `evaluateAndReport()`

这意味着无论底层算法是神经网络、线性模型、树模型还是样本记忆模型，外部调用方式都保持一致。

`ModelFactory` 负责根据模型类型枚举创建实例，当前支持：

- `SVM`
- `FCNN`
- `CNN`
- `LOGISTIC_REGRESSION`
- `RANDOM_FOREST`
- `KNN`

## 3. 数据流与训练测试流程

### 3.1 数据来源

项目使用 MNIST 数据集，读取方式定义在 `Models.cpp` 中：

- `readImages()`
- `readLabels()`

训练集和测试集在 Qt 后端中统一加载：

- 训练集：
  - `train-images-idx3-ubyte`
  - `train-labels-idx1-ubyte`
- 测试集：
  - `t10k-images-idx3-ubyte`
  - `t10k-labels-idx1-ubyte`

### 3.2 输入预处理

所有图像在读取时都被转换成 `double`，并做如下归一化：

`pixel = unsigned_char / 255.0`

也就是统一缩放到 `[0, 1]` 区间。

注意：

- 这里有输入归一化，但没有使用 `BatchNorm`、`LayerNorm` 等网络层归一化模块。
- 正则化主要出现在部分模型的损失或参数更新项中，而不是以独立层的形式出现。

### 3.3 统一测试与评估流程

所有模型测试时都走统一流程：

1. 加载参数文件
2. 对测试集逐样本预测
3. 计算 `Accuracy`
4. 计算混淆矩阵 `Confusion Matrix`
5. 计算每类的 `Precision / Recall / F1`
6. 计算宏平均和微平均指标
7. 计算 `ROC / AUC`

其中：

- 混淆矩阵和 Accuracy 基于统一的 `predict()` 接口
- AUC 基于 `predictProba()` 输出的类别概率做 one-vs-rest 计算
- ROC 文件会保存到 `ROC/*.txt`

## 4. 各算法的详细结构

下面按“前向结构 + 训练方式 + 特点”来整理每个模型。

### 4.1 SVM

模型说明：

- 线性多分类 SVM
- 使用 one-vs-rest 方式处理 10 类数字分类
- 使用 hinge loss
- 使用 mini-batch SGD 优化

前向结构：

`Input(784) -> normalize([0,1]) -> 10个线性分类头 -> score(10) -> argmax`

更细化地写：

`input -> affine(W_c, b_c) for each class -> class scores -> argmax`

训练结构：

`Input(784) -> Linear One-vs-Rest Heads(10) -> Hinge Loss -> Gradient + L2-style regularization -> Mini-batch SGD`

训练细节：

- 输入为 784 维展平图像向量
- 每个类别维护一组 `weights + bias`
- 每个类别单独按二分类方式构造标签：
  - 当前类为 `+1`
  - 其他类为 `-1`
- 损失函数为：
  - `max(0, 1 - y * score)`
- 权重梯度中加入正则项
- 每个 batch 先累计梯度，再统一更新

说明：

- 预测类别由 10 个线性 score 中最大值决定
- `predictProba()` 中会把 score 做 softmax 归一化，主要用于 AUC/ROC 计算
- 本质上它仍然是线性 SVM，而不是 softmax 分类器

### 4.2 FCNN / BP 神经网络

模型说明：

- 标准全连接前馈神经网络
- 使用反向传播，属于典型 BP 神经网络
- 两层隐藏层
- 激活函数为 `ReLU`
- 输出层使用 `Softmax`

前向结构：

`Input(784) -> normalize([0,1]) -> Dense(256) -> ReLU -> Dense(128) -> ReLU -> Dense(10) -> Softmax`

也可以写成更标准的层表达：

`input -> affine(W1,b1) -> ReLU -> affine(W2,b2) -> ReLU -> affine(W3,b3) -> Softmax`

训练结构：

`Input(784) -> Dense(256) -> ReLU -> Dense(128) -> ReLU -> Dense(10) -> Softmax -> CrossEntropy -> Backpropagation -> SGD + Momentum`

参数规模：

- 输入层：`784`
- 隐藏层 1：`256`
- 隐藏层 2：`128`
- 输出层：`10`

训练细节：

- 权重初始化：`He initialization`
- 损失函数：`cross entropy`
- 优化器：`SGD + Momentum`
- 动量更新形式：
  - `v = mu * v + grad`
  - `w = w - lr * v`

反向传播路径：

1. 输出层计算
   - `dz3 = softmax_output - one_hot(label)`
2. 更新 `W3, b3`
3. 反传到第二隐藏层
4. 乘上 ReLU 导数，得到 `dz2`
5. 更新 `W2, b2`
6. 继续反传到第一隐藏层
7. 再乘上 ReLU 导数，得到 `dz1`
8. 更新 `W1, b1`

说明：

- 项目中的 BP 神经网络没有使用 Dropout
- 没有使用 BatchNorm
- 没有显式 L2 正则项

### 4.3 CNN

模型说明：

- 手写实现的卷积神经网络
- 不依赖深度学习框架
- 包含两层卷积、两层池化和一个全连接输出层

前向结构：

`Input(28x28) -> normalize([0,1]) -> Conv(8,3x3) -> ReLU -> MaxPool(2x2) -> Conv(16,3x3) -> ReLU -> MaxPool(2x2) -> Flatten -> Dense(10) -> Softmax`

按尺寸展开：

1. 输入：
   - `28 x 28 x 1`
2. 第一层卷积：
   - `Conv(8, 3x3)`
   - 输出：`26 x 26 x 8`
3. 第一层池化：
   - `MaxPool(2x2)`
   - 输出：`13 x 13 x 8`
4. 第二层卷积：
   - `Conv(16, 3x3)`
   - 输出：`11 x 11 x 16`
5. 第二层池化：
   - `MaxPool(2x2)`
   - 输出：`5 x 5 x 16`
6. 展平：
   - `Flatten = 16 * 5 * 5 = 400`
7. 输出层：
   - `Dense(10)`
8. 概率输出：
   - `Softmax`

训练结构：

`Input(28x28) -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Dense(10) -> Softmax -> CrossEntropy -> Backprop`

训练细节：

- 卷积核初始化使用 He 风格初始化
- 第一卷积层：
  - 8 个 `3x3` 卷积核
- 第二卷积层：
  - 16 个输出通道
  - 每个输出通道连接前一层 8 个输入通道
- 最大池化会记录最大值位置，用于反向传播
- 输出层梯度：
  - `dLogits = prob - one_hot`
- 之后依次做：
  - 全连接层反向传播
  - `unflatten`
  - 第二层池化反向传播
  - 第二层 ReLU 反向传播
  - 第二层卷积核更新
  - 第一层池化反向传播
  - 第一层 ReLU 反向传播
  - 第一层卷积核更新

说明：

- 代码中的 CNN 只有一个最终分类全连接层，即 `Flatten -> Dense(10)`
- 没有额外的隐藏全连接层
- 没有 BatchNorm、Dropout、残差连接等现代结构
- 虽然外层按 batch 遍历数据，但参数更新是在样本级别逐个执行的，属于偏在线更新风格

### 4.4 Logistic Regression

模型说明：

- 单层 softmax 回归
- 本质是多分类线性分类器

前向结构：

`Input(784) -> normalize([0,1]) -> Dense(10) -> Softmax`

标准表达：

`input -> affine(W,b) -> logits(10) -> Softmax`

训练结构：

`Input(784) -> Linear Projection -> Softmax -> CrossEntropy + L2 -> Mini-batch SGD`

训练细节：

- 权重初始化：`Xavier initialization`
- 损失：`softmax cross entropy`
- 正则化：`L2`
- 优化器：`mini-batch SGD`

梯度形式：

- `gradient = probs - one_hot(label)`

说明：

- 没有隐藏层
- 没有非线性激活层
- 决策边界是线性的
- 可以视为 FCNN 的最简化版本

### 4.5 Random Forest

模型说明：

- 随机森林集成模型
- 由多棵决策树组成
- 不是神经网络，因此没有 `affine / relu / softmax hidden stack` 这类层级结构

整体结构：

`Input(784) -> Bootstrap Sampling -> Decision Tree Ensemble(50 trees) -> Voting / Probability Average`

更细流程：

1. 输入为 784 维归一化像素特征
2. 使用 bootstrap 抽样构建每棵树的数据子集
3. 节点分裂时随机抽取一部分特征
4. 用 Gini impurity 选择最佳划分
5. 递归生长到停止条件
6. 多棵树的结果做投票或概率平均

当前实现参数：

- 树数量：`50`
- 最大深度：`8`
- 最小分裂样本数：`8`
- 每次分裂考虑的特征数：`28`

节点结构包括：

- `featureIndex`
- `threshold`
- `predictedClass`
- `leftChild`
- `rightChild`
- `classCounts`

停止分裂条件：

- 深度达到上限
- 样本数过少
- 当前节点已经纯净

预测方式：

- `predict()`
  - 所有树分别预测类别
  - 多数投票
- `predictProba()`
  - 每棵树在叶节点给出类别分布
  - 对所有树的类别概率做平均

说明：

- `epochs` 参数对随机森林没有真正意义
- 代码中明确写明：随机森林不是按 epoch 优化，而是一次性建立固定数量的树

### 4.6 KNN

模型说明：

- 精确 K 近邻分类器
- 基于样本存储和距离度量
- 属于非参数模型

整体结构：

`Input(784) -> Sample Memory -> k-Nearest Search(k=5) -> Distance-Weighted Vote`

更细流程：

1. 输入为 784 维归一化像素特征
2. 训练阶段把训练样本及标签全部存入内存
3. 预测阶段与所有存储样本计算距离
4. 找到最近的 `k=5` 个邻居
5. 对邻居做距离加权投票
6. 输出类别或归一化后的类别概率

距离定义：

- 使用平方欧氏距离

投票权重：

- `1 / (sqrt(distance) + 1e-6)`

预测方式：

- `predict()`
  - 对最近邻做加权投票
- `predictProba()`
  - 将各类别投票权重归一化后作为概率输出

说明：

- KNN 的“训练”本质上不是参数优化，而是样本存储
- `epochs` 参数在 KNN 中也没有真实的迭代优化含义
- 当前 `train()` 中实际会把训练样本全部存入 `prototypes`

## 5. 超参数概览

当前项目在 `Models.cpp` 中定义的主要超参数如下：

### SVM

- `learning rate = 1e-3`
- `regularization C = 0.01`
- `epochs = 5`
- `batch size = 64`

### FCNN

- `learning rate = 1e-3`
- `momentum = 0.9`
- `epochs = 5`
- `batch size = 64`
- `hidden1 = 256`
- `hidden2 = 128`

### CNN

- `learning rate = 1e-3`
- `epochs = 15`
- `batch size = 64`

### Logistic Regression

- `learning rate = 1e-3`
- `L2 = 0.0001`
- `epochs = 5`
- `batch size = 64`

### Random Forest

- `trees = 50`
- `max depth = 8`
- `min samples split = 8`
- `features per split = 28`

### KNN

- `k = 5`
- `max prototypes = 3000`

注意：

- `Random Forest` 和 `KNN` 虽然保留了统一训练接口中的 `epochs` 参数，但实际并不依赖 epoch 做迭代优化。

## 6. 命令行与图形界面

### 6.1 命令行界面

命令行入口位于：

- `cli_main.cpp`

功能包括：

- 选择模型
- 训练模型
- 测试模型
- 查看测试指标

构建完成后可运行：

```powershell
.\build-qt\build\cnn_cli.exe
```

### 6.2 Qt 图形界面

图形界面入口位于：

- `build-qt/gui-src/main.cpp`

主要页面包括：

- `Train`
  - 选择模型
  - 设置 epoch
  - 控制是否从已有参数继续训练
  - 实时显示训练进度
- `Test`
  - 加载参数文件
  - 输出 Accuracy、Precision、Recall、F1、AUC
  - 展示 ROC 曲线、混淆矩阵、分类别指标
- `Visualize`
  - 展示每种模型的结构示意图
  - 提供结构说明和阶段说明

启动方式：

```powershell
.\Mathine Learning.cmd
```

## 7. 构建方式

Qt 应用的构建目录在：

- `build-qt/gui-src/`

构建脚本会生成两个可执行目标：

- `cnn_gui`
- `cnn_cli`

依赖：

- Qt 6
- Qt Widgets
- Qt Charts
- Qt Concurrent
- C++17

## 8. 总结

这个项目的核心价值在于：把多种常见机器学习算法封装到一个统一的 C++ 实验框架内，使它们可以在同一数据集、同一接口、同一评估体系下进行对比。

如果从结构角度概括，最重要的三个神经网络/线性模型可以记为：

- `SVM`
  - `Input(784) -> Linear OvR Heads(10) -> Hinge Loss`
- `FCNN / BP`
  - `Input(784) -> Dense(256) -> ReLU -> Dense(128) -> ReLU -> Dense(10) -> Softmax`
- `CNN`
  - `Input(28x28) -> Conv(8,3x3) -> ReLU -> MaxPool -> Conv(16,3x3) -> ReLU -> MaxPool -> Flatten -> Dense(10) -> Softmax`
- `Logistic Regression`
  - `Input(784) -> Dense(10) -> Softmax`
- `Random Forest`
  - `Input(784) -> 50 Decision Trees -> Voting / Probability Average`
- `KNN`
  - `Input(784) -> Sample Memory -> k=5 Nearest Search -> Weighted Vote`

这份 README 基于当前仓库中的真实代码实现整理，而不是基于抽象教材结构描述，因此可直接作为本项目的结构说明文档使用。
