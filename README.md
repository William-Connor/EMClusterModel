# 🎯 EMClusterModel

**基于 GMM-EM 算法的模式识别聚类系统**

从零实现高斯混合模型（Gaussian Mixture Model）与期望最大化（Expectation-Maximization）算法，应用于手写数字自动聚类。

## 📊 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行实验
python scripts/run_experiments.py --dataset pendigits --k 10 --pca 12

# 启动 Web 交互界面
streamlit run app.py
```

## 📁 项目结构

```
EMClusterModel/
├── data/                    # 原始与处理后数据
├── src/
│   ├── data_loader.py       # 数据加载（UCI Pen Digits / Optdigits）
│   ├── preprocess.py        # 预处理流水线（标准化、PCA、划分）
│   ├── em_gmm.py            # GMM-EM 核心实现（从零手写，~370行）
│   ├── evaluation.py        # 评估指标（匈牙利匹配、P/R/F1、ARI、NMI）
│   └── visualize.py         # 可视化（散点图、混淆矩阵、错误分析等）
├── scripts/
│   └── run_experiments.py   # 实验运行脚本
├── app.py                   # Streamlit 交互界面
├── report/
│   ├── REPORT.md            # 完整实验报告
│   └── figures/             # 生成的图表
└── requirements.txt
```

## 🧠 算法说明

**GMM（高斯混合模型）**：假设数据由 K 个高斯分布混合生成，每个数据点以概率 π_k 来自第 k 分量。

**EM 算法**迭代估计参数：
- **E-step**：计算每个样本属于各分量的后验概率（责任度 γ）
- **M-step**：基于责任度加权更新均值 μ_k、协方差 Σ_k、混合系数 π_k

支持四种协方差类型：`full` | `tied` | `diag` | `spherical`

## 📈 实验数据集

| 数据集 | 样本数 | 维度 | 类别 | 来源 |
|--------|--------|------|------|------|
| Pen Digits | 10,992 | 16 | 10 | UCI |
| Optdigits | 5,620 | 64 | 10 | UCI |
| sklearn Digits | 1,797 | 64 | 10 | sklearn |

## 🎯 实验结论

- Pen Digits：Accuracy 67.9%, F1 0.671, ARI 0.572
- sklearn Digits：Accuracy 22.6% — 展示高维小样本下的局限性
- 完整分析见 [实验报告](report/REPORT.md)

## 📝 License

MIT
