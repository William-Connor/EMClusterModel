#!/usr/bin/env python3
"""
EM 聚类算法工程 — 主实验脚本
运行全套实验：加载数据 → 预处理 → 模型选择 → 训练 → 评估 → 可视化

用法: python scripts/run_experiments.py
      python scripts/run_experiments.py --dataset pendigits --k 10 --pca 30
"""

import os
import sys
import argparse
import json
import warnings
import time

warnings.filterwarnings("ignore")

# 将项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.data_loader import get_all_datasets
from src.preprocess import preprocess_pipeline, apply_pca
from src.em_gmm import GaussianMixtureEM, em_model_selection
from src.evaluation import compute_all_metrics, print_evaluation
from src.visualize import (
    plot_clustering_comparison, plot_confusion_matrix,
    plot_per_class_metrics, plot_model_selection,
    plot_em_convergence, plot_error_cases, plot_overall_summary,
    REPORT_DIR
)

RANDOM_STATE = 42


def run_single_experiment(dataset_name, X, y, k=None, pca_dim=None, cov_type="full"):
    """
    对单个数据集运行完整实验流水线
    
    参数:
        dataset_name: 数据集名称
        X, y: 特征与标签
        k: 聚类数（None 则自动选择）
        pca_dim: PCA 降维维度（None 则不降维）
        cov_type: 协方差类型
    """
    print(f"\n{'='*70}")
    print(f"  实验: {dataset_name}")
    print(f"{'='*70}")
    
    n_classes = len(np.unique(y))
    
    # === 1. 预处理 ===
    print("\n--- Step 1: 数据预处理 ---")
    data = preprocess_pipeline(X, y, test_size=0.3, pca_components=pca_dim,
                                random_state=RANDOM_STATE)
    print(f"  训练集: {data['X_train'].shape}, 测试集: {data['X_test'].shape}")
    
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    # === 2. 模型选择 (确定最优 K) ===
    if k is None:
        print("\n--- Step 2: 模型选择 (BIC/AIC) ---")
        k_range = list(range(2, min(n_classes + 8, 21)))
        sel_results = em_model_selection(X_train, k_range, covariance_type=cov_type,
                                          n_init=3, random_state=RANDOM_STATE)
        # 选 BIC 最小的 K
        k = sel_results["k"][np.argmin(sel_results["bic"])]
        print(f"  BIC 最优 K = {k}")
        
        # 保存模型选择图
        plot_model_selection(sel_results,
                             save_path=f"{dataset_name}_model_selection.png",
                             show=False)
    else:
        sel_results = None
        print(f"\n--- Step 2: 使用指定 K = {k} ---")
    
    # === 3. 训练 GMM-EM ===
    print(f"\n--- Step 3: 训练 GMM-EM (K={k}, cov={cov_type}) ---")
    t0 = time.time()
    gmm = GaussianMixtureEM(
        n_components=k, covariance_type=cov_type,
        max_iter=300, tol=1e-3, n_init=3,
        random_state=RANDOM_STATE, verbose=True
    )
    gmm.fit(X_train)
    train_time = time.time() - t0
    print(f"  训练耗时: {train_time:.2f}s, 迭代: {gmm.n_iter_}, 收敛: {gmm.converged_}")
    
    # 收敛曲线
    plot_em_convergence(gmm.history_,
                        title=f"EM Convergence - {dataset_name}",
                        save_path=f"{dataset_name}_convergence.png",
                        show=False)
    
    # === 4. 预测 ===
    print("\n--- Step 4: 预测 ---")
    y_pred_train = gmm.predict(X_train)
    y_pred_test = gmm.predict(X_test)
    
    # === 5. 评估 ===
    print("\n--- Step 5: 评估 ---")
    results_train = compute_all_metrics(y_train, y_pred_train, X_train)
    results_test = compute_all_metrics(y_test, y_pred_test, X_test)
    
    print("\n[训练集评估]")
    print_evaluation(results_train)
    print("\n[测试集评估]")
    print_evaluation(results_test)
    
    # VIP: y_pred 对齐后的标签
    from src.evaluation import hungarian_align
    y_pred_aligned_test, _ = hungarian_align(y_test, y_pred_test)
    
    # === 6. 可视化 ===
    print("\n--- Step 6: 生成可视化图表 ---")
    
    # 聚类对比散点图
    plot_clustering_comparison(
        X_test, y_test, y_pred_test,
        title=f"Clustering Result - {dataset_name} (Test Set)",
        save_path=f"{dataset_name}_comparison.png", show=False
    )
    
    # 混淆矩阵
    plot_confusion_matrix(
        results_test["confusion"],
        classes=sorted(set(y_test)),
        title=f"Confusion Matrix - {dataset_name}",
        save_path=f"{dataset_name}_confusion.png", show=False
    )
    
    # 各类别指标
    plot_per_class_metrics(
        results_test["per_class"],
        title=f"Per-Class Metrics - {dataset_name}",
        save_path=f"{dataset_name}_per_class.png", show=False
    )
    
    # 错误案例分析
    plot_error_cases(
        X_test, y_test, y_pred_test, y_pred_aligned_test,
        title=f"Error Case Analysis - {dataset_name}",
        save_path=f"{dataset_name}_error_cases.png", show=False
    )
    
    # 综合指标
    plot_overall_summary(
        results_test,
        title=f"Performance Summary - {dataset_name}",
        save_path=f"{dataset_name}_summary.png", show=False
    )
    
    # === 7. 边界讨论数据收集 ===
    error_mask = (y_test != y_pred_aligned_test)
    error_indices = np.where(error_mask)[0]
    error_details = []
    for idx in error_indices[:20]:  # 最多记录 20 个错误案例
        error_details.append({
            "index": int(idx),
            "true_label": int(y_test[idx]),
            "predicted_label": int(y_pred_aligned_test[idx]),
            "raw_prediction": int(y_pred_test[idx]),
            "probability": gmm.predict_proba(X_test[idx:idx+1])[0].tolist()
        })
    
    # === 汇总 ===
    summary = {
        "dataset": dataset_name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "pca_dim": pca_dim,
        "selected_k": k,
        "covariance_type": cov_type,
        "train_time_seconds": round(train_time, 2),
        "n_iterations": gmm.n_iter_,
        "converged": gmm.converged_,
        "final_log_likelihood": round(gmm.lower_bound_, 2),
        "bic": round(gmm.bic(X_test), 2),
        "aic": round(gmm.aic(X_test), 2),
        "train_metrics": {
            "accuracy": round(results_train["accuracy"], 4),
            "precision_macro": round(results_train["precision_macro"], 4),
            "recall_macro": round(results_train["recall_macro"], 4),
            "f1_macro": round(results_train["f1_macro"], 4),
            "ari": round(results_train["ari"], 4),
            "nmi": round(results_train["nmi"], 4),
        },
        "test_metrics": {
            "accuracy": round(results_test["accuracy"], 4),
            "precision_macro": round(results_test["precision_macro"], 4),
            "recall_macro": round(results_test["recall_macro"], 4),
            "f1_macro": round(results_test["f1_macro"], 4),
            "precision_weighted": round(results_test["precision_weighted"], 4),
            "recall_weighted": round(results_test["recall_weighted"], 4),
            "f1_weighted": round(results_test["f1_weighted"], 4),
            "ari": round(results_test["ari"], 4),
            "nmi": round(results_test["nmi"], 4),
        },
        "n_errors_test": int(error_mask.sum()),
        "error_rate": round(float(error_mask.sum()) / len(y_test), 4),
        "error_details": error_details,
        "model_selection": {
            "k_candidates": sel_results["k"] if sel_results else [],
            "bic_values": sel_results["bic"] if sel_results else [],
            "aic_values": sel_results["aic"] if sel_results else [],
        } if sel_results else None,
    }
    
    # 保存汇总
    summary_path = os.path.join(REPORT_DIR, f"{dataset_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="EM Clustering Experiment Runner")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "pendigits", "optdigits", "digits"],
                        help="数据集选择 (默认: all)")
    parser.add_argument("--k", type=int, default=None,
                        help="聚类数 (默认: 自动选择)")
    parser.add_argument("--pca", type=int, default=None,
                        help="PCA 降维维度 (默认: 不降维)")
    parser.add_argument("--cov", type=str, default="full",
                        choices=["full", "tied", "diag", "spherical"],
                        help="协方差类型 (默认: full)")
    args = parser.parse_args()
    
    # 加载数据
    all_datasets = get_all_datasets()
    if not all_datasets:
        print("[ERROR] 没有可用的数据集！请检查网络连接。")
        sys.exit(1)
    
    # 筛选
    if args.dataset != "all":
        if args.dataset in all_datasets:
            all_datasets = {args.dataset: all_datasets[args.dataset]}
        else:
            print(f"[ERROR] 数据集 '{args.dataset}' 不可用")
            print(f"  可用: {list(all_datasets.keys())}")
            sys.exit(1)
    
    # 为每个数据集选择合适的 PCA 维度
    pca_config = {
        "pendigits": args.pca if args.pca else 12,
        "optdigits": args.pca if args.pca else 30,
        "digits": args.pca if args.pca else None,
    }
    
    all_summaries = {}
    n_classes = None  # 自动检测
    
    for name, (X, y) in all_datasets.items():
        n_classes = len(np.unique(y))
        pca = pca_config.get(name)
        k_target = args.k if args.k else n_classes  # 默认使用真实类别数
        
        summary = run_single_experiment(
            name, X, y, k=k_target, pca_dim=pca, cov_type=args.cov
        )
        all_summaries[name] = summary
    
    # 保存总汇总
    total_path = os.path.join(REPORT_DIR, "all_summaries.json")
    with open(total_path, "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}")
    print(f"  全部实验完成！结果已保存至 {REPORT_DIR}")
    print(f"{'='*70}")
    
    # 打印总览表
    print("\n📊 实验结果总览:\n")
    print(f"{'数据集':<15} {'K':>3} {'PCA':>5} {'Acc':>7} {'F1_m':>7} {'ARI':>7} {'NMI':>7} {'Err%':>7}")
    print("-" * 65)
    for name, s in all_summaries.items():
        tm = s["test_metrics"]
        print(f"{name:<15} {s['selected_k']:>3} {s.get('pca_dim') or '-':>5} "
              f"{tm['accuracy']:>7.3f} {tm['f1_macro']:>7.3f} "
              f"{tm['ari']:>7.3f} {tm['nmi']:>7.3f} {s['error_rate']:>6.1%}")


if __name__ == "__main__":
    main()
