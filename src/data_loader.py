"""
EM 聚类算法工程 — 数据加载模块
支持数据集：Pen Digits (UCI), Optdigits (UCI), Digits (sklearn)
"""

import os
import numpy as np
from sklearn.datasets import load_digits
from ucimlrepo import fetch_ucirepo


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def load_pendigits():
    """
    加载 Pen-Based Recognition of Handwritten Digits 数据集
    来源: UCI ML Repository (ID: 81)
    样本: 10,992 (训练 7,494 + 测试 3,498)
    特征: 16 (笔迹坐标采样)
    类别: 10 (数字 0-9)
    """
    print("[INFO] 正在加载 Pen Digits 数据集...")
    pendigits = fetch_ucirepo(id=81)
    X = pendigits.data.features.values.astype(np.float64)
    y = pendigits.data.targets.values.ravel().astype(np.int64)
    print(f"[OK] Pen Digits: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类")
    return X, y


def load_optdigits():
    """
    加载 Optical Recognition of Handwritten Digits 数据集
    来源: UCI ML Repository (ID: 80)
    样本: 5,620 (训练 3,823 + 测试 1,797)
    特征: 64 (8x8 像素灰度值)
    类别: 10 (数字 0-9)
    """
    print("[INFO] 正在加载 Optdigits 数据集...")
    optdigits = fetch_ucirepo(id=80)
    X = optdigits.data.features.values.astype(np.float64)
    y = optdigits.data.targets.values.ravel().astype(np.int64)
    print(f"[OK] Optdigits: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类")
    return X, y


def load_sklearn_digits():
    """
    加载 sklearn 内置 Digits 数据集
    样本: 1,797
    特征: 64 (8x8 像素灰度值)
    类别: 10 (数字 0-9)
    """
    print("[INFO] 正在加载 sklearn Digits 数据集...")
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)
    print(f"[OK] sklearn Digits: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类")
    return X, y


def get_all_datasets():
    """
    返回所有可用数据集
    返回: dict[str, tuple[X, y]]
    """
    datasets = {}
    try:
        X, y = load_pendigits()
        datasets["pendigits"] = (X, y)
    except Exception as e:
        print(f"[WARN] Pen Digits 加载失败: {e}")
    try:
        X, y = load_optdigits()
        datasets["optdigits"] = (X, y)
    except Exception as e:
        print(f"[WARN] Optdigits 加载失败: {e}")
    try:
        X, y = load_sklearn_digits()
        datasets["digits"] = (X, y)
    except Exception as e:
        print(f"[WARN] sklearn Digits 加载失败: {e}")
    return datasets


if __name__ == "__main__":
    ds = get_all_datasets()
    for name, (X, y) in ds.items():
        print(f"\n=== {name} ===")
        print(f"  形状: X={X.shape}, y={y.shape}")
        print(f"  类别分布: {np.bincount(y.astype(int))}")
        print(f"  X 范围: [{X.min():.2f}, {X.max():.2f}]")
