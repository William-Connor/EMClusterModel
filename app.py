"""
EM 聚类算法工程 — Streamlit 交互界面
提供数据集选择、参数调节、训练、评估、可视化的全功能 Web 界面
"""

import sys
import os
import time

# 确保项目路径正确
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.data_loader import get_all_datasets
from src.preprocess import preprocess_pipeline
from src.em_gmm import GaussianMixtureEM
from src.evaluation import compute_all_metrics, hungarian_align
from src.visualize import (
    plot_clustering_comparison, plot_confusion_matrix,
    plot_per_class_metrics, plot_em_convergence,
    plot_error_cases, plot_overall_summary
)

# 页面配置
st.set_page_config(
    page_title="EM Clustering Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 标题
st.title("🎯 EM 聚类算法 — 模式识别演示系统")
st.markdown("""
基于高斯混合模型（GMM）与期望最大化（EM）算法的无监督聚类系统。
使用手写数字数据集（Pen Digits / Optdigits），对多维笔迹/像素特征进行聚类。
""")

# === 侧边栏：参数配置 ===
st.sidebar.header("⚙️ 参数配置")

# 数据集选择
dataset_option = st.sidebar.selectbox(
    "📦 数据集",
    options=["pendigits", "optdigits", "digits", "mnist", "fashion_mnist"],
    format_func=lambda x: {
        "pendigits": "Pen Digits (10,992 × 16)",
        "optdigits": "Optdigits (5,620 × 64)",
        "digits": "sklearn Digits (1,797 × 64)",
        "mnist": "MNIST (70,000 × 784)",
        "fashion_mnist": "Fashion-MNIST (70,000 × 784)",
    }[x],
    help="选择用于聚类实验的数据集"
)

# K 值
use_auto_k = st.sidebar.checkbox("自动选择 K (BIC/AIC)", value=True)
if use_auto_k:
    k_value = None
else:
    k_value = st.sidebar.slider("聚类数 K", min_value=2, max_value=20, value=10)

# PCA 降维
use_pca = st.sidebar.checkbox("PCA 降维", value=True)
if use_pca:
    pca_dim = st.sidebar.slider("PCA 维度", min_value=2, max_value=64, value=30)
else:
    pca_dim = None

# 协方差类型
cov_type = st.sidebar.selectbox(
    "协方差类型",
    options=["full", "tied", "diag", "spherical"],
    index=0,
    help="full: 完整协方差 | tied: 共享协方差 | diag: 对角 | spherical: 球面"
)

# 高级参数
with st.sidebar.expander("🔧 高级参数"):
    max_iter = st.slider("最大迭代次数", 50, 500, 300, 50)
    tol = st.select_slider("收敛阈值", options=[1e-2, 1e-3, 1e-4, 1e-5], value=1e-3,
                           format_func=lambda x: f"{x:.0e}")
    n_init = st.slider("随机初始化次数", 1, 10, 3)

# === 主界面 ===
# 运行按钮
if st.sidebar.button("🚀 运行实验", type="primary", use_container_width=True):
    run_experiment = True
else:
    run_experiment = False

if run_experiment:
    # Step 1: 加载数据
    with st.spinner("正在加载数据..."):
        all_ds = get_all_datasets()
        if dataset_option not in all_ds:
            st.error(f"数据集 {dataset_option} 加载失败，请检查网络连接。")
            st.stop()
        X, y = all_ds[dataset_option]
    
    st.success(f"✅ 数据集加载成功：{X.shape[0]} 样本 × {X.shape[1]} 特征，{len(np.unique(y))} 类")
    
    # Step 2: 预处理
    with st.spinner("数据预处理中..."):
        data = preprocess_pipeline(X, y, test_size=0.3, pca_components=pca_dim)
    
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("训练集", f"{X_train.shape[0]} 样本")
    with col2:
        st.metric("测试集", f"{X_test.shape[0]} 样本")
    
    # Step 3: 训练
    n_classes = len(np.unique(y))
    actual_k = k_value or n_classes
    
    st.markdown(f"⏳ **训练中** (K={actual_k}, cov={cov_type})...")
    progress_bar = st.progress(0, text="初始化...")
    status_text = st.empty()
    
    # 进度回调：init_idx(0..n_init-1), iteration(0..max_iter-1), max_iter
    def on_progress(init_idx, iteration, max_iter):
        total_iters = n_init * max_iter
        completed = init_idx * max_iter + (iteration + 1)
        pct = min(completed / total_iters, 0.99)
        progress_bar.progress(pct, text=f"初始化 {init_idx+1}/{n_init} · 迭代 {iteration+1}/{max_iter}")
        status_text.caption(f"⏳ EM 迭代中... 第 {init_idx+1} 次初始化")
    
    t0 = time.time()
    gmm = GaussianMixtureEM(
        n_components=actual_k, covariance_type=cov_type,
        max_iter=max_iter, tol=tol, n_init=n_init,
        random_state=42, verbose=False,
        progress_callback=on_progress
    )
    gmm.fit(X_train)
    train_time = time.time() - t0
    
    progress_bar.progress(1.0, text="训练完成！")
    status_text.empty()
    st.success(f"✅ 训练完成！耗时 {train_time:.2f}s，迭代 {gmm.n_iter_} 次，"
                f"{'已收敛' if gmm.converged_ else '达到最大迭代'}")
    
    # Step 4: 预测与评估
    y_pred_test = gmm.predict(X_test)
    y_pred_aligned, mapping = hungarian_align(y_test, y_pred_test)
    results = compute_all_metrics(y_test, y_pred_test, X_test)
    
    # === 结果展示 ===
    st.header("📊 实验结果")
    
    # 指标卡片行
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Accuracy", f"{results['accuracy']:.3f}")
    with c2:
        st.metric("F1 (macro)", f"{results['f1_macro']:.3f}")
    with c3:
        st.metric("Precision", f"{results['precision_macro']:.3f}")
    with c4:
        st.metric("Recall", f"{results['recall_macro']:.3f}")
    with c5:
        st.metric("ARI", f"{results['ari']:.3f}")
    with c6:
        st.metric("NMI", f"{results['nmi']:.3f}")
    
    # Tab 页切换不同视图
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 聚类对比", "📋 评估详情", "📈 各类别分析", "⚠️ 错误分析", "🔄 收敛曲线"
    ])
    
    with tab1:
        st.subheader("真实标签 vs 聚类结果")
        fig = plot_clustering_comparison(
            X_test, y_test, y_pred_test,
            title=f"Clustering Result - {dataset_option}",
            show=False
        )
        st.pyplot(fig)
    
    with tab2:
        st.subheader("评估指标详情")
        
        # 混淆矩阵
        fig_cm = plot_confusion_matrix(
            results["confusion"],
            classes=sorted(set(y_test)),
            title=f"Confusion Matrix - {dataset_option}",
            show=False
        )
        st.pyplot(fig_cm)
        
        # 标签映射
        st.write("**标签映射 (聚类 → 真实类别):**")
        st.json(mapping)
        
        # 综合指标
        fig_sum = plot_overall_summary(
            results,
            title="Performance Summary",
            show=False
        )
        st.pyplot(fig_sum)
    
    with tab3:
        st.subheader("各类别 P/R/F1")
        st.dataframe(
            results["per_class"].set_index("class").style.format("{:.3f}"),
            use_container_width=True
        )
        fig_pc = plot_per_class_metrics(
            results["per_class"],
            title=f"Per-Class Metrics - {dataset_option}",
            show=False
        )
        st.pyplot(fig_pc)
    
    with tab4:
        st.subheader("错误案例分析")
        errors = (y_test != y_pred_aligned)
        n_errors = errors.sum()
        error_rate = n_errors / len(y_test) * 100
        
        st.metric("错误数", f"{n_errors} / {len(y_test)} ({error_rate:.2f}%)")
        
        fig_err = plot_error_cases(
            X_test, y_test, y_pred_test, y_pred_aligned,
            title=f"Error Analysis - {dataset_option}",
            show=False
        )
        st.pyplot(fig_err)
        
        if n_errors > 0:
            st.write("**错误案例详情（前20条）：**")
            proba = gmm.predict_proba(X_test)
            error_data = []
            for idx in np.where(errors)[0][:20]:
                error_data.append({
                    "索引": int(idx),
                    "真实类别": int(y_test[idx]),
                    "聚类分配": int(y_pred_aligned[idx]),
                    "原始预测": int(y_pred_test[idx]),
                    "最大概率": f"{proba[idx].max():.3f}",
                    "概率分布": str(np.round(proba[idx], 2).tolist())
                })
            st.dataframe(pd.DataFrame(error_data), use_container_width=True)
    
    with tab5:
        st.subheader("EM 收敛曲线")
        fig_conv = plot_em_convergence(
            gmm.history_,
            title=f"EM Convergence - {dataset_option}",
            show=False
        )
        st.pyplot(fig_conv)
    
    # BIC/AIC
    st.header("📐 模型复杂度分析")
    col_bic, col_aic = st.columns(2)
    with col_bic:
        st.metric("BIC", f"{gmm.bic(X_test):.2f}")
    with col_aic:
        st.metric("AIC", f"{gmm.aic(X_test):.2f}")

else:
    # 未运行时显示说明
    st.info("👈 请在左侧配置参数，然后点击「运行实验」按钮开始。")
    
    st.markdown("""
    ### 📖 使用说明
    
    1. **选择数据集**：五种数据集可选
       - Pen Digits: 笔迹坐标特征，16 维
       - Optdigits: 像素灰度特征，64 维  
       - sklearn Digits: 小规模，64 维
       - MNIST: 大规模手写数字，784 维
       - Fashion-MNIST: 大规模服饰图像，784 维
    2. **配置参数**：可选择自动确定 K 或手动指定
    3. **运行实验**：点击按钮开始训练
    4. **查看结果**：在多个 Tab 页中查看聚类效果
    
    ### 🧠 算法简介
    
    **GMM（高斯混合模型）** 假设数据由 K 个高斯分布混合生成，每个数据点属于各分布的概率由后验概率确定。
    
    **EM 算法**迭代求解：
    - **E-step**：基于当前参数，估计每个点属于各分布的责任度 γ
    - **M-step**：基于责任度，重新最大化似然来更新均值、协方差和混合系数
    
    迭代至对数似然收敛，得到最终聚类结果。
    """)

# 页脚
st.divider()
st.caption("EMClusterModel · GMM-EM 聚类算法工程 · 模式识别应用")
