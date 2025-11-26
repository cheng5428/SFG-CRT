# SFG-复现：CTR预测的生成式范式

本仓库包含了论文 *[From Feature Interaction to Feature Generation: A Generative Paradigm of CTR Prediction Models (ICML 2025)]* 中提出的**监督特征生成（SFG）**框架的复现代码。

本项目基于 [ReChorus](https://github.com/THUwangcy/ReChorus) 框架实现。

## 📄 论文信息
> **From Feature Interaction to Feature Generation: A Generative Paradigm of CTR Prediction Models**  
> Mingjia Yin, Junwei Pan, Hao Wang, et al.  
> *ICML 2025*

**核心思想：**
从判别式的"特征交互"范式转向生成式的"特征生成"范式，使用**All-Predict-All**策略来缓解嵌入维度崩塌和信息冗余问题。

## 🚀 已实现模型
我们使用SFG编码器复现了以下模型：
1. **GenFM**：将SFG应用于因子分解机（Factorization Machines）
2. **GenDeepFM**：将SFG应用于DeepFM

关键文件位置：
- `src/models/GenFM.py`
- `src/models/GenDeepFM.py`

## 🛠️ 环境要求

```bash
git clone https://github.com/cheng5428/SFG-CRT.git
cd SFG-CRT
pip install -r requirements.txt
```

## 📊 数据集准备

### MovieLens-1M 数据集

对于 **MovieLens-1M** 数据集，只需运行预处理脚本即可：

```bash
cd data/MovieLens_1M
python MovieLens-1M-CTR.py
```

该脚本将自动完成：
- 下载 MovieLens-1M 数据集
- 解压并处理数据
- 在 `./ML_1MCTR/` 目录下生成 CTR 预测格式的文件

### MIND-Large 数据集

对于 **MIND-Large** 数据集，需要手动下载：

1. **下载数据集**，从 [Microsoft News Dataset (MIND)](https://msnews.github.io/) 下载：
   - `MINDlarge_train.zip`
   - `MINDlarge_dev.zip`

2. **创建目录结构**：
   ```bash
   cd data/MIND_Large
   mkdir -p MIND_large
   ```

3. **放置下载的文件**：
   ```
   data/MIND_Large/
   └── MIND_large/
       ├── MINDlarge_train.zip
       └── MINDlarge_dev.zip
   ```

4. **运行预处理脚本**：
   ```bash
   python MIND-large-CTR.py
   ```

该脚本将完成：
- 解压zip文件
- 执行5-core过滤（保证用户和物品至少有5次交互）
- 生成基于时间的特征（小时、星期、时段等）
- 划分训练集/验证集/测试集
- 将处理后的数据保存到 `./MINDCTR/` 目录

**注意**：MIND-Large 数据集较大，请确保有足够的磁盘空间和内存用于处理。

## 🏃‍♂️ 运行模型

准备好数据集后，可以训练模型：

训练 **GenFM** 模型：

```bash
bash run_GenFM_MIND.bash    # 在MIND数据集上训练
bash run_GenFM_ML1M.bash    # 在MovieLens-1M数据集上训练
```

训练 **GenDeepFM** 模型：

```bash
bash run_GenDeepFM_MIND.bash    # 在MIND数据集上训练
bash run_GenDeepFM_ML1M.bash    # 在MovieLens-1M数据集上训练
```

一键完成所有模型的训练：
```bash
bash run_all.bash
```

## 📊 评估指标

本项目使用以下指标评估CTR预测性能：
- **AUC**
- **Log Loss**
