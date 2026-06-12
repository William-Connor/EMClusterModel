"""
EM 聚类算法工程 — 数据预处理模块
功能：标准化、PCA降维、数据划分
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def standardize_data(X_train, X_test=None):
    """
    Z-score 标准化：使每个特征均值为 0，标准差为 1
    
    为什么需要标准化？
    EM 算法使用高斯分布建模，若不同特征量纲差异大（如特征 A 范围 [0,1]，特征 B 范围 [0,1000]），
    协方差矩阵会被量纲大的特征主导，导致聚类结果偏向该特征。
    标准化后各特征平等贡献。
    
    例子：
    >>> X = np.array([[1, 100], [2, 200], [3, 300]])
    >>> X_scaled, _ = standardize_data(X)
    >>> X_scaled  # 两列均变为均值 0，标准差 1
    array([[-1.2247, -1.2247],
           [ 0.    ,  0.    ],
           [ 1.2247,  1.2247]])
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


def apply_pca(X_train, X_test=None, n_components=2, verbose=True):
    """
    PCA 降维：保留主要方差方向，减少噪声
    
    为什么需要 PCA？
    高维数据（如 Optdigits 64 维）直接聚类时，欧氏距离在高维空间中会失效（维度灾难）。
    EM 算法估算的高斯协方差矩阵在样本不足时可能奇异。PCA 去噪 + 降维后效果更好。
    
    例子：
    >>> X = np.random.randn(100, 64)  # 64 维数据
    >>> X_pca, _ = apply_pca(X, n_components=2)
    >>> X_pca.shape
    (100, 2)
    """
    if verbose:
        explained_before = PCA().fit(X_train).explained_variance_ratio_
        cumsum = np.cumsum(explained_before)
        n95 = np.searchsorted(cumsum, 0.95) + 1
        print(f"[PCA] 原始维度: {X_train.shape[1]}, 保留 95% 方差需 {n95} 维")
    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    
    if verbose:
        print(f"[PCA] 降维至 {n_components} 维, 方差保留率: {pca.explained_variance_ratio_.sum():.4f}")
    
    if X_test is not None:
        X_test_pca = pca.transform(X_test)
        return X_train_pca, X_test_pca, pca
    return X_train_pca, pca


def split_data(X, y, test_size=0.3, random_state=42):
    """
    划分训练集/测试集。分层抽样保证各类别比例一致。
    
    例子：
    >>> X = np.random.randn(100, 5)
    >>> y = np.array([0]*50 + [1]*50)
    >>> X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.3)
    >>> len(X_tr), len(X_te)
    (70, 30)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def preprocess_pipeline(X, y, test_size=0.3, pca_components=None, random_state=42):
    """
    完整预处理流水线
    1. 划分训练/测试集
    2. 标准化（在训练集上 fit，测试集上 transform）
    3. 可选 PCA 降维
    
    参数:
        pca_components: PCA 目标维度，None 表示不降维
    
    返回:
        dict: {
            "X_train_raw": ..., "X_test_raw": ...,
            "X_train": ..., "X_test": ...,
            "y_train": ..., "y_test": ...,
            "scaler": ..., "pca": ...
        }
    """
    # 1. 划分
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 2. 标准化
    X_train_scaled, X_test_scaled, scaler = standardize_data(X_train_raw, X_test_raw)
    
    # 3. PCA
    pca_model = None
    X_train_final, X_test_final = X_train_scaled, X_test_scaled
    if pca_components is not None:
        X_train_final, X_test_final, pca_model = apply_pca(
            X_train_scaled, X_test_scaled, n_components=pca_components
        )
    
    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train": X_train_final,
        "X_test": X_test_final,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "pca": pca_model,
    }


if __name__ == "__main__":
    from data_loader import load_optdigits
    X, y = load_optdigits()
    result = preprocess_pipeline(X, y, pca_components=30)
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}")
