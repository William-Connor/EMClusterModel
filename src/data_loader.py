"""
EM 聚类算法工程 — 数据加载模块
支持数据集：Pen Digits (UCI), Optdigits (UCI), Digits (sklearn)

策略：优先从本地 data/raw/ 加载 .npy 缓存，仅首次运行时从网络下载。
"""

import os
import numpy as np
from sklearn.datasets import load_digits

# 数据目录：项目根目录下的 data/raw/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def _load_or_download(name, download_fn):
    """
    通用加载器：优先本地缓存，缺失时下载并缓存
    
    参数:
        name: 数据集名称（用于文件名）
        download_fn: 无参函数，返回 (X, y) 的下载函数
    返回:
        X, y: numpy arrays
    """
    x_path = os.path.join(DATA_DIR, f"{name}_X.npy")
    y_path = os.path.join(DATA_DIR, f"{name}_y.npy")
    
    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"[INFO] 从本地加载 {name} 数据集...")
        X = np.load(x_path).astype(np.float64)
        y = np.load(y_path).astype(np.int64)
        print(f"[OK] {name}: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类 (本地缓存)")
        return X, y
    
    print(f"[INFO] 本地缓存缺失，从网络下载 {name} 数据集...")
    X, y = download_fn()
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"[OK] {name}: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类 (已缓存)")
    return X, y


def _download_pendigits():
    """从 UCI 下载 Pen Digits"""
    from ucimlrepo import fetch_ucirepo
    pendigits = fetch_ucirepo(id=81)
    X = pendigits.data.features.values.astype(np.float64)
    y = pendigits.data.targets.values.ravel().astype(np.int64)
    return X, y


def _download_optdigits():
    """从 UCI 下载 Optdigits"""
    from ucimlrepo import fetch_ucirepo
    optdigits = fetch_ucirepo(id=80)
    X = optdigits.data.features.values.astype(np.float64)
    y = optdigits.data.targets.values.ravel().astype(np.int64)
    return X, y


def _download_sklearn_digits():
    """从 sklearn 加载 Digits"""
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(np.int64)
    return X, y


def load_pendigits():
    """
    加载 Pen-Based Recognition of Handwritten Digits 数据集
    样本: 10,992 | 特征: 16 | 类别: 10
    优先使用 data/raw/pendigits_X.npy + pendigits_y.npy
    """
    return _load_or_download("pendigits", _download_pendigits)


def load_optdigits():
    """
    加载 Optical Recognition of Handwritten Digits 数据集
    样本: 5,620 | 特征: 64 | 类别: 10
    优先使用 data/raw/optdigits_X.npy + optdigits_y.npy
    """
    return _load_or_download("optdigits", _download_optdigits)


def load_sklearn_digits():
    """
    加载 sklearn 内置 Digits 数据集
    样本: 1,797 | 特征: 64 | 类别: 10
    优先使用 data/raw/digits_X.npy + digits_y.npy
    """
    return _load_or_download("digits", _download_sklearn_digits)


def get_all_datasets():
    """
    返回所有可用数据集
    返回: dict[str, tuple[X, y]]
    """
    datasets = {}
    for name, loader in [
        ("pendigits", load_pendigits),
        ("optdigits", load_optdigits),
        ("digits", load_sklearn_digits),
    ]:
        try:
            X, y = loader()
            datasets[name] = (X, y)
        except Exception as e:
            print(f"[WARN] {name} 加载失败: {e}")
    return datasets


if __name__ == "__main__":
    ds = get_all_datasets()
    for name, (X, y) in ds.items():
        print(f"\n=== {name} ===")
        print(f"  形状: X={X.shape}, y={y.shape}")
        print(f"  类别分布: {np.bincount(y.astype(int))}")
        print(f"  X 范围: [{X.min():.2f}, {X.max():.2f}]")
