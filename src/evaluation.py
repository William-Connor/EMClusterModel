"""
EM 聚类算法工程 — 评估指标模块
包含：匈牙利匹配、Accuracy/Precision/Recall/F1、ARI、NMI、轮廓系数等
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    adjusted_rand_score, normalized_mutual_info_score,
    confusion_matrix, silhouette_score
)
import pandas as pd


def hungarian_align(y_true, y_pred):
    """
    匈牙利算法：寻找聚类标签 → 真实标签的最优匹配
    
    EM 聚类产生的标签索引（0,1,2...）是任意的，和真实数字标签可能不对应。
    匈牙利算法通过最大化匹配的样本数，为每个聚类标签找到最可能的真实类别。
    
    例子:
    >>> y_true = [0, 0, 1, 1, 2, 2]
    >>> y_pred = [2, 2, 0, 0, 1, 1]  # 聚类标签偏移了
    >>> hungarian_align(y_true, y_pred)
    array([0, 0, 1, 1, 2, 2])  # 正确对齐
    
    返回:
        y_pred_aligned: 对齐后的预测标签
        mapping: dict {聚类标签 → 真实标签}
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    n_classes = max(y_true.max(), y_pred.max()) + 1
    # 构建代价矩阵：cost[i, j] = -(聚类标签 j 中属于真实类别 i 的样本数)
    cost = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n_classes):
        mask_true = (y_true == i)
        for j in range(n_classes):
            mask_pred = (y_pred == j)
            cost[i, j] = -np.sum(mask_true & mask_pred)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {col: row for row, col in zip(row_ind, col_ind) if row < n_classes and col < n_classes}
    
    y_aligned = np.array([mapping.get(p, -1) for p in y_pred])
    return y_aligned, mapping


def compute_all_metrics(y_true, y_pred, X=None):
    """
    计算全套聚类评估指标
    
    参数:
        y_true: 真实标签
        y_pred: 聚类预测标签
        X:      (可选) 原始特征矩阵，用于轮廓系数
    
    返回:
        dict: 包含所有指标
    """
    # 匈牙利对齐
    y_aligned, mapping = hungarian_align(y_true, y_pred)
    valid_mask = y_aligned >= 0
    
    results = {
        "mapping": mapping,
        "accuracy": accuracy_score(y_true[valid_mask], y_aligned[valid_mask]),
        "precision_macro": precision_score(y_true[valid_mask], y_aligned[valid_mask], average="macro", zero_division=0),
        "recall_macro": recall_score(y_true[valid_mask], y_aligned[valid_mask], average="macro", zero_division=0),
        "f1_macro": f1_score(y_true[valid_mask], y_aligned[valid_mask], average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true[valid_mask], y_aligned[valid_mask], average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true[valid_mask], y_aligned[valid_mask], average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true[valid_mask], y_aligned[valid_mask], average="weighted", zero_division=0),
        "ari": adjusted_rand_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
    }
    
    if X is not None:
        try:
            results["silhouette"] = silhouette_score(X, y_pred)
        except Exception:
            results["silhouette"] = None
    
    # 每个类别的详细指标
    unique_classes = sorted(set(y_true[valid_mask]))
    class_metrics = []
    for c in unique_classes:
        yt = (y_true == c).astype(int)
        yp = (y_aligned == c).astype(int)
        # 找到 y_aligned 中是否有这个值
        if np.sum(yp) == 0:
            class_metrics.append({
                "class": c, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "support": np.sum(yt)
            })
        else:
            class_metrics.append({
                "class": c,
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "support": int(np.sum(yt))
            })
    results["per_class"] = pd.DataFrame(class_metrics)
    
    # 混淆矩阵
    results["confusion"] = confusion_matrix(y_true[valid_mask], y_aligned[valid_mask])
    
    return results


def print_evaluation(results):
    """格式化打印评估结果"""
    print("\n" + "="*60)
    print("                    聚类评估结果")
    print("="*60)
    print(f"  Accuracy:              {results['accuracy']:.4f}")
    print(f"  Precision (macro):     {results['precision_macro']:.4f}")
    print(f"  Recall (macro):        {results['recall_macro']:.4f}")
    print(f"  F1-score (macro):      {results['f1_macro']:.4f}")
    print(f"  Precision (weighted):  {results['precision_weighted']:.4f}")
    print(f"  Recall (weighted):     {results['recall_weighted']:.4f}")
    print(f"  F1-score (weighted):   {results['f1_weighted']:.4f}")
    print(f"  ARI (调整兰德指数):     {results['ari']:.4f}")
    print(f"  NMI (归一化互信息):     {results['nmi']:.4f}")
    if results.get("silhouette") is not None:
        print(f"  Silhouette Score:      {results['silhouette']:.4f}")
    print(f"  Mapping:               {results['mapping']}")
    print("="*60)
    print("\n各分类详细指标:")
    print(results["per_class"].to_string(index=False))
    return results


if __name__ == "__main__":
    # 快速测试
    y_true = np.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 3])
    y_pred = np.array([2, 2, 0, 0, 1, 1, 1, 3, 3, 3])
    r = compute_all_metrics(y_true, y_pred)
    print_evaluation(r)
