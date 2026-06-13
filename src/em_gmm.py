"""
EM 聚类算法工程 — GMM-EM 核心实现
从零实现高斯混合模型 + EM（期望最大化）算法
"""

import numpy as np
from scipy.stats import multivariate_normal
from collections import defaultdict


class GaussianMixtureEM:
    """
    从零实现的高斯混合模型（Gaussian Mixture Model），使用 EM 算法进行参数估计。
    
    算法基本思想
    ============
    假设数据由 K 个高斯分布混合生成，每个数据点以概率 π_k 来自第 k 个高斯分布。
    GMM 的目标是估计参数 θ = {π_k, μ_k, Σ_k}，使得观测数据的似然最大。
    
    由于每个数据点属于哪个分布是隐变量（未知），无法直接最大化似然。
    EM 算法迭代解决：
    
    E-step (Expectation): 基于当前参数估算每个点属于各分布的后验概率（责任度 γ）
    M-step (Maximization): 基于责任度重新估计参数（加权最大似然）
    
    迭代至对数似然收敛。
    
    协方差类型支持
    ==============
    - "full":       每个分量独立完整协方差矩阵  Σ_k ∈ R^{d×d}
    - "tied":       所有分量共享一个协方差矩阵  Σ
    - "diag":       每个分量对角协方差         Σ_k = diag(σ_k1², ..., σ_kd²)
    - "spherical":  每个分量球面协方差           Σ_k = σ_k²·I
    """
    
    def __init__(self, n_components=3, covariance_type="full", max_iter=200,
                 tol=1e-4, n_init=5, random_state=None, verbose=False,
                 progress_callback=None):
        """
        参数:
            n_components:   高斯分量数 K
            covariance_type: 协方差类型 "full" / "tied" / "diag" / "spherical"
            max_iter:       EM 最大迭代次数
            tol:            对数似然收敛阈值
            n_init:         随机初始化次数（取最优）
            random_state:   随机种子
            verbose:        是否打印迭代信息
            progress_callback: 进度回调函数 callback(init_idx, iteration, max_iter)
                              用于 Streamlit 进度条等场景
        """
        assert covariance_type in ("full", "tied", "diag", "spherical")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.progress_callback = progress_callback
        
        # 训练后填充
        self.weights_ = None       # 混合系数 π_k, shape (K,)
        self.means_ = None         # 均值 μ_k, shape (K, d)
        self.covariances_ = None   # 协方差 Σ_k, 类型取决于 covariance_type
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -np.inf  # 对数似然
        self.history_ = []         # 每次迭代的对数似然记录
    
    def _initialize_parameters(self, X, rng):
        """K-Means++ 初始化均值，协方差初始化为数据方差"""
        n_samples, n_features = X.shape
        K = self.n_components
        
        # K-Means++ 选初始均值
        means = np.zeros((K, n_features))
        # 随机选第一个中心
        first_idx = rng.randint(n_samples)
        means[0] = X[first_idx].copy()
        
        for k in range(1, K):
            # 计算每个点到最近中心的距离平方
            dist_sq = np.array([
                np.min([np.sum((X[i] - means[j])**2) for j in range(k)])
                for i in range(n_samples)
            ])
            probs = dist_sq / dist_sq.sum()
            means[k] = X[rng.choice(n_samples, p=probs)].copy()
        
        # 混合系数初始均匀
        weights = np.ones(K) / K
        
        # 协方差初始化：全局协方差
        global_cov = np.cov(X.T, bias=True)
        if global_cov.ndim == 0:
            global_cov = np.array([[global_cov]])
        
        if self.covariance_type == "full":
            covs = np.array([global_cov.copy() for _ in range(K)])
            # 加小正则项防止奇异
            for k in range(K):
                covs[k] += 1e-6 * np.eye(n_features)
        elif self.covariance_type == "tied":
            covs = global_cov.copy()
            covs += 1e-6 * np.eye(n_features)
        elif self.covariance_type == "diag":
            covs = np.array([np.diag(global_cov).copy() for _ in range(K)])
            covs = np.maximum(covs, 1e-6)
        elif self.covariance_type == "spherical":
            covs = np.array([np.trace(global_cov) / n_features for _ in range(K)])
            covs = np.maximum(covs, 1e-6)
        
        return weights, means, covs
    
    def _estimate_log_gaussian_prob(self, X, means, covs):
        """
        计算每个样本在每个高斯分量下的对数概率密度
        
        E-step 核心：log p(x_i | z_i=k, θ)
        
        参数:
            X: (n_samples, n_features)
            means: (K, n_features)
            covs: 协方差，类型取决于 covariance_type
        
        返回:
            log_prob: (n_samples, K) 对数概率
        """
        n_samples, n_features = X.shape
        K = self.n_components
        log_prob = np.zeros((n_samples, K))
        
        for k in range(K):
            mean = means[k]
            if self.covariance_type == "full":
                cov = covs[k]
            elif self.covariance_type == "tied":
                cov = covs  # 所有分量共用
            elif self.covariance_type == "diag":
                cov = np.diag(covs[k])
            elif self.covariance_type == "spherical":
                cov = covs[k] * np.eye(n_features)
            
            # 使用 scipy 计算多元高斯对数概率
            try:
                log_prob[:, k] = multivariate_normal.logpdf(X, mean=mean, cov=cov, allow_singular=True)
            except Exception:
                # 如果协方差奇异，使用伪逆
                cov_reg = cov + 1e-6 * np.eye(n_features)
                log_prob[:, k] = multivariate_normal.logpdf(X, mean=mean, cov=cov_reg, allow_singular=True)
        
        return log_prob
    
    def _e_step(self, X, log_weights, means, covs):
        """
        E-step: 计算责任度（后验概率）
        
        γ(z_{ik}) = p(z_i=k | x_i, θ) = π_k * N(x_i | μ_k, Σ_k) / Σ_j[π_j * N(x_i | μ_j, Σ_j)]
        
        返回:
            log_resp: (n_samples, K) 对数责任度
            log_prob_norm: (n_samples,) 对数似然（每个样本的归一化常数）
        """
        # log π_k + log p(x_i | z_i=k)
        log_prob = self._estimate_log_gaussian_prob(X, means, covs)
        weighted_log_prob = log_prob + log_weights  # (n_samples, K)
        
        # log-sum-exp trick 防止高维数值下溢导致 NaN
        max_w = np.max(weighted_log_prob, axis=1, keepdims=True)
        log_prob_norm = (max_w + np.log(
            np.sum(np.exp(weighted_log_prob - max_w), axis=1, keepdims=True)
        )).ravel()
        
        # log γ(z_{ik}) = weighted - log(norm)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        
        return log_resp, log_prob_norm
    
    def _m_step(self, X, log_resp):
        """
        M-step: 基于责任度更新参数
        """
        n_samples, n_features = X.shape
        K = self.n_components
        
        # γ(z_{ik}) = exp(log_resp)
        resp = np.exp(log_resp)
        Nk = resp.sum(axis=0) + 1e-12  # 每个分量的有效样本数 (K,)
        
        # 1. 更新混合系数 π_k = Nk / N
        weights = Nk / n_samples
        
        # 2. 更新均值 μ_k = (1/Nk) Σ γ(z_{ik}) * x_i
        means = (resp.T @ X) / Nk[:, np.newaxis]
        
        # 3. 更新协方差
        if self.covariance_type == "tied":
            # Σ = (1/N) Σ_k Σ_i γ_{ik} (x_i - μ_k)(x_i - μ_k)^T
            cov = np.zeros((n_features, n_features))
            for k in range(K):
                diff = X - means[k]
                cov += (resp[:, k][:, np.newaxis] * diff).T @ diff
            cov /= n_samples
            cov += 1e-6 * np.eye(n_features)
            covs = cov
        
        elif self.covariance_type == "full":
            # Σ_k = (1/Nk) Σ_i γ_{ik} (x_i - μ_k)(x_i - μ_k)^T
            covs = np.zeros((K, n_features, n_features))
            for k in range(K):
                diff = X - means[k]
                covs[k] = (resp[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]
                covs[k] += 1e-6 * np.eye(n_features)
        
        elif self.covariance_type == "diag":
            # Σ_k = diag( (1/Nk) Σ_i γ_{ik} (x_i - μ_k)² )
            covs = np.zeros((K, n_features))
            for k in range(K):
                diff = X - means[k]
                covs[k] = np.sum(resp[:, k][:, np.newaxis] * diff**2, axis=0) / Nk[k]
                covs[k] = np.maximum(covs[k], 1e-6)
        
        elif self.covariance_type == "spherical":
            # σ_k² = (1/(d*Nk)) Σ_i γ_{ik} ||x_i - μ_k||²
            covs = np.zeros(K)
            for k in range(K):
                diff = X - means[k]
                covs[k] = np.sum(resp[:, k] * np.sum(diff**2, axis=1)) / (n_features * Nk[k])
                covs[k] = max(covs[k], 1e-6)
        
        return weights, means, covs
    
    def _compute_lower_bound(self, X, log_resp, log_prob_norm):
        """
        计算完整的对数似然下界（Evidence Lower Bound / ELBO）
        """
        return np.sum(log_prob_norm)
    
    def _fit_single(self, X, rng, init_idx=0):
        """单次 EM 拟合"""
        weights, means, covs = self._initialize_parameters(X, rng)
        log_weights = np.log(weights + 1e-12)
        prev_lower_bound = -np.inf
        history = []
        
        for it in range(self.max_iter):
            log_resp, log_prob_norm = self._e_step(X, log_weights, means, covs)
            weights, means, covs = self._m_step(X, log_resp)
            log_weights = np.log(weights + 1e-12)
            
            lower_bound = self._compute_lower_bound(X, log_resp, log_prob_norm)
            history.append(lower_bound)
            
            # 进度回调
            if self.progress_callback is not None:
                self.progress_callback(init_idx, it, self.max_iter)
            
            if self.verbose and it % 10 == 0:
                print(f"  [EM] iter {it:4d}  log-likelihood = {lower_bound:.4f}")
            
            if it > 0 and abs(lower_bound - prev_lower_bound) < self.tol:
                if self.verbose:
                    print(f"  [EM] 收敛于 iter {it}, log-likelihood = {lower_bound:.4f}")
                return weights, means, covs, lower_bound, True, it + 1, history
            
            prev_lower_bound = lower_bound
        
        if self.verbose:
            print(f"  [EM] 达到最大迭代 {self.max_iter}, log-likelihood = {prev_lower_bound:.4f}")
        return weights, means, covs, prev_lower_bound, False, self.max_iter, history
    
    def fit(self, X):
        """
        拟合 GMM 模型。多次随机初始化，取对数似然最大的一次。
        
        参数:
            X: (n_samples, n_features) 训练数据
        
        返回:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        assert n_samples >= self.n_components, f"样本数 {n_samples} < 分量数 {self.n_components}"
        
        if self.random_state is not None:
            base_rng = np.random.RandomState(self.random_state)
        else:
            base_rng = np.random.RandomState()
        
        best_lower_bound = -np.inf
        best_params = None
        
        for init_i in range(self.n_init):
            if self.verbose:
                print(f"[INIT] 第 {init_i + 1}/{self.n_init} 次初始化")
            
            rng = np.random.RandomState(base_rng.randint(2**31 - 1))
            weights, means, covs, lb, converged, it, history = self._fit_single(X, rng, init_idx=init_i)
            
            if lb > best_lower_bound:
                best_lower_bound = lb
                best_params = (weights, means, covs, converged, it, history)
        
        self.weights_, self.means_, self.covariances_, self.converged_, self.n_iter_, self.history_ = best_params
        self.lower_bound_ = best_lower_bound
        
        if self.verbose:
            print(f"[FINAL] 最优对数似然: {best_lower_bound:.4f}, 收敛: {self.converged_}, 迭代: {self.n_iter_}")
        
        return self
    
    def predict(self, X):
        """
        硬分配：返回每个样本最可能属于的高斯分量
        
        返回:
            labels: (n_samples,) 分量索引 [0, K-1]
        """
        log_weights = np.log(self.weights_ + 1e-12)
        log_prob = self._estimate_log_gaussian_prob(X, self.means_, self.covariances_)
        weighted = log_prob + log_weights
        return np.argmax(weighted, axis=1)
    
    def predict_proba(self, X):
        """
        软分配：返回每个样本属于各分量的后验概率
        使用 log-sum-exp 技巧防止数值溢出（修复 Infinity bug）
        
        返回:
            resp: (n_samples, K) 责任度矩阵
        """
        log_weights = np.log(self.weights_ + 1e-12)
        log_prob = self._estimate_log_gaussian_prob(X, self.means_, self.covariances_)
        weighted = log_prob + log_weights
        # log-sum-exp: log(Σexp(x_j)) = max_k + log(Σexp(x_j - max_k))
        max_w = np.max(weighted, axis=1, keepdims=True)
        log_prob_norm = max_w + np.log(np.sum(np.exp(weighted - max_w), axis=1, keepdims=True))
        return np.exp(weighted - log_prob_norm)
    
    def score_samples(self, X):
        """返回每个样本的对数似然（用于异常检测），使用 log-sum-exp 防止溢出"""
        log_weights = np.log(self.weights_ + 1e-12)
        log_prob = self._estimate_log_gaussian_prob(X, self.means_, self.covariances_)
        weighted = log_prob + log_weights
        # log-sum-exp trick: log(Σexp(x_i)) = max(x) + log(Σexp(x_i - max(x)))
        max_w = np.max(weighted, axis=1, keepdims=True)
        return (max_w + np.log(np.sum(np.exp(weighted - max_w), axis=1))).ravel()
    
    def score(self, X):
        """返回全体数据的平均对数似然"""
        return np.mean(self.score_samples(X))
    
    def bic(self, X):
        """
        BIC (Bayesian Information Criterion)
        BIC = -2 * log_likelihood + n_params * log(n_samples)
        值越小越好，用于模型选择（确定最优 K）
        """
        n_samples, n_features = X.shape
        K = self.n_components
        
        # 自由参数数量
        # 均值: K * d
        n_params = K * n_features
        # 混合系数: K - 1 (因为 Σπ_k = 1)
        n_params += K - 1
        # 协方差
        if self.covariance_type == "full":
            n_params += K * n_features * (n_features + 1) / 2
        elif self.covariance_type == "tied":
            n_params += n_features * (n_features + 1) / 2
        elif self.covariance_type == "diag":
            n_params += K * n_features
        elif self.covariance_type == "spherical":
            n_params += K
        
        log_likelihood = self.score(X) * n_samples
        return -2 * log_likelihood + n_params * np.log(n_samples)
    
    def aic(self, X):
        """
        AIC (Akaike Information Criterion)
        AIC = -2 * log_likelihood + 2 * n_params
        """
        n_samples, n_features = X.shape
        K = self.n_components
        
        n_params = K * n_features + (K - 1)
        if self.covariance_type == "full":
            n_params += K * n_features * (n_features + 1) / 2
        elif self.covariance_type == "tied":
            n_params += n_features * (n_features + 1) / 2
        elif self.covariance_type == "diag":
            n_params += K * n_features
        elif self.covariance_type == "spherical":
            n_params += K
        
        log_likelihood = self.score(X) * n_samples
        return -2 * log_likelihood + 2 * n_params


def em_model_selection(X, k_range, covariance_type="full", n_init=3, random_state=42):
    """
    通过 BIC/AIC 选择最优 K 值
    返回每个 K 对应的 BIC/AIC/对数似然
    """
    results = {"k": [], "bic": [], "aic": [], "log_likelihood": []}
    for k in k_range:
        print(f"  [Model Selection] K={k}...")
        gmm = GaussianMixtureEM(
            n_components=k, covariance_type=covariance_type,
            max_iter=200, tol=1e-4, n_init=n_init,
            random_state=random_state, verbose=False
        )
        gmm.fit(X)
        results["k"].append(k)
        results["bic"].append(gmm.bic(X))
        results["aic"].append(gmm.aic(X))
        results["log_likelihood"].append(gmm.score(X) * X.shape[0])
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from src.data_loader import load_sklearn_digits
    from src.preprocess import preprocess_pipeline
    
    X, y = load_sklearn_digits()
    data = preprocess_pipeline(X, y, pca_components=30)
    
    print("\n=== 测试 GMM-EM ===")
    gmm = GaussianMixtureEM(n_components=10, covariance_type="full", n_init=3, verbose=True)
    gmm.fit(data["X_train"])
    
    pred = gmm.predict(data["X_test"])
    print(f"预测标签: {np.bincount(pred)}")
    print(f"BIC: {gmm.bic(data['X_test']):.2f}")
