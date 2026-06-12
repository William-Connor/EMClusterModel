"""
EM 聚类算法工程 — 可视化模块
功能：降维散点图、混淆矩阵、评估指标柱状图、模型选择曲线等
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免 GUI 依赖
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 中文字体设置（尝试多种方案）
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

sns.set_style("whitegrid")
sns.set_palette("husl")

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "report", "figures")
os.makedirs(REPORT_DIR, exist_ok=True)


def plot_clustering_comparison(X, y_true, y_pred, title="Clustering Comparison",
                                save_path=None, show=True):
    """
    并排展示真实标签 vs 聚类结果的 PCA 降维散点图
    
    这是评估聚类质量最直观的方式——把高维数据降到 2D，
    左边看"正确答案"，右边看"算法输出"，肉眼看差异。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PCA 降维至 2D
    if X.shape[1] > 2:
        X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        X_2d = X
    
    # 左侧：真实标签
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap="tab10",
                                alpha=0.7, s=15, edgecolors="none")
    axes[0].set_title("Ground Truth Labels", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    legend1 = axes[0].legend(*scatter1.legend_elements(), title="Class",
                               loc="upper right", markerscale=2)
    axes[0].add_artist(legend1)
    
    # 右侧：聚类结果
    scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap="tab10",
                                alpha=0.7, s=15, edgecolors="none")
    axes[1].set_title("EM Clustering Result", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    legend2 = axes[1].legend(*scatter2.legend_elements(), title="Cluster",
                               loc="upper right", markerscale=2)
    axes[1].add_artist(legend2)
    
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(REPORT_DIR, save_path)
        plt.savefig(full_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {full_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig


def plot_confusion_matrix(cm, classes=None, title="Confusion Matrix",
                           save_path=None, show=True):
    """
    混淆矩阵热力图
    行 = 真实标签，列 = 聚类对齐后标签
    对角线越深越好（聚类正确匹配）
    """
    if classes is None:
        classes = list(range(cm.shape[0]))
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                xticklabels=classes, yticklabels=classes,
                cbar_kws={"label": "Number of Samples"})
    ax.set_xlabel("Predicted Cluster (aligned)", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(REPORT_DIR, save_path)
        plt.savefig(full_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {full_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig


def plot_per_class_metrics(per_class_df, title="Per-Class Metrics",
                            save_path=None, show=True):
    """
    每个类别的 Precision / Recall / F1 柱状图
    论文标准格式
    """
    classes = per_class_df["class"].tolist()
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width, per_class_df["precision"], width, label="Precision", color="#2E86AB")
    bars2 = ax.bar(x, per_class_df["recall"], width, label="Recall", color="#A23B72")
    bars3 = ax.bar(x + width, per_class_df["f1"], width, label="F1-Score", color="#F18F01")
    
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)
    
    # 在柱子上标注数值
    for bar in bars1 + bars2 + bars3:
        height = bar.get_height()
        if height > 0.05:
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(REPORT_DIR, save_path)
        plt.savefig(full_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {full_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig


def plot_model_selection(results, save_path=None, show=True):
    """
    模型选择曲线：BIC + AIC + Log-Likelihood vs K
    帮助确定最优分量数
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    k = results["k"]
    
    # BIC
    axes[0].plot(k, results["bic"], "o-", color="#2E86AB", linewidth=2, markersize=8)
    axes[0].set_title("BIC (lower is better)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Components K")
    axes[0].set_ylabel("BIC")
    bic_best = k[np.argmin(results["bic"])]
    axes[0].axvline(x=bic_best, color="red", linestyle="--", alpha=0.7, label=f"Best K={bic_best}")
    axes[0].legend()
    
    # AIC
    axes[1].plot(k, results["aic"], "s-", color="#A23B72", linewidth=2, markersize=8)
    axes[1].set_title("AIC (lower is better)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Components K")
    axes[1].set_ylabel("AIC")
    aic_best = k[np.argmin(results["aic"])]
    axes[1].axvline(x=aic_best, color="red", linestyle="--", alpha=0.7, label=f"Best K={aic_best}")
    axes[1].legend()
    
    # Log-Likelihood
    axes[2].plot(k, results["log_likelihood"], "D-", color="#F18F01", linewidth=2, markersize=8)
    axes[2].set_title("Log-Likelihood (higher is better)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Number of Components K")
    axes[2].set_ylabel("Log-Likelihood")
    
    fig.suptitle("GMM Model Selection", fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(REPORT_DIR, save_path)
        plt.savefig(full_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {full_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig


def plot_em_convergence(history, title="EM Convergence", save_path=None, show=True):
    """
    EM 算法收敛曲线：对数似然随迭代的变化
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    iterations = list(range(len(history)))
    ax.plot(iterations, history, "b-", linewidth=2)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Log-Likelihood", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(y=history[-1], color="red", linestyle="--", alpha=0.5,
               label=f"Final = {history[-1]:.2f}")
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(REPORT_DIR, save_path)
        plt.savefig(full_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {full_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig


def plot_error_cases(X, y_true, y_pred, y_aligned, title="Error Case Analysis",
                     save_path=None, show=True):
    """
    错误案例分析：突出显示聚类错误的样本
    绿色=聚类正确，红色=聚类错误
    """
    if X.shape[1] > 2:
        X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        X_2d = X
    
    errors = (y_true != y_aligned)
    correct = ~errors
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 全图
    axes[0].scatter(X_2d[correct, 0], X_2d[correct, 1], c="green", alpha=0.3, s=10, label="Correct")
    axes[0].scatter(X_2d[errors, 0], X_2d[errors, 1], c="red", alpha=0.6, s=20, label="Wrong", edgecolors="black", linewidth=0.5)
    axes[0].set_title(f"Error Distribution ({errors.sum()}/{len(y_true)} errors)", fontsize=12, fontweight="bold")
    axes[0].legend()
    
    # 放大错误区域（仅显示错误点，按真实类别着色）
    if errors.sum() > 0:
        scatter = axes[1].scatter(X_2d[errors, 0], X_2d[errors, 1],
                                   c=y_true[errors], cmap="tab10", alpha=0.8, s=25, edgecolors="black", linewidth=0.5)
        # 标注每个点所属的真实类别
        unique_err_classes = np.unique(y_true[errors])
        handles = []
        for cls in unique_err_classes:
            mask = y_true[errors] == cls
            handles.append(mpatches.Patch(color=scatter.cmap(scatter.norm(cls)), label=f"True: {cls}"))
        axes[1].legend(handles=handles, title="True Class", fontsize=8, title_fontsize=9)
    axes[1].set_title("Error Samples by True Class", fontsize=12, fontweight="bold")
    
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(REPORT_DIR, save_path)
        plt.savefig(full_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {full_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig


def plot_overall_summary(results, title="Clustering Performance Summary",
                          save_path=None, show=True):
    """综合指标仪表盘"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                    "precision_weighted", "recall_weighted", "f1_weighted", "ari", "nmi"]
    metrics_labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)",
                       "Precision\n(weighted)", "Recall\n(weighted)", "F1\n(weighted)", "ARI", "NMI"]
    values = [results[k] for k in metrics_keys]
    
    colors = sns.color_palette("viridis", len(values))
    bars = ax.barh(metrics_labels, values, color=colors)
    
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                va="center", fontsize=10, fontweight="bold")
    
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axvline(x=0.9, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(REPORT_DIR, save_path)
        plt.savefig(full_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {full_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig


if __name__ == "__main__":
    # 测试
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=300, centers=4, n_features=10, random_state=42)
    y_pred = np.random.randint(0, 4, 300)
    plot_clustering_comparison(X, y, y_pred, save_path="test_comparison.png", show=False)
    print("Visualization module OK")
