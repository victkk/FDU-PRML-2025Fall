#!/usr/bin/env python3
"""
实验：类内方差（CLUSTER_STD）对 kNN 最优 k 值的影响

研究问题：
1. CLUSTER_STD ↑（更模糊）时，best_k 是否趋向更大？
2. 为什么从"锯齿→平滑"的边界有助于抗噪？
3. 对比 k=1 与 k=best_k 的误分类点分布，哪些区域最难？
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from data_generate import generate_and_save, load_prepared_dataset
from knn_student import select_k_by_validation, knn_predict

# 实验配置
CLUSTER_STDS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]  # 类内标准差范围
KS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]             # k值候选
N_SAMPLES = 500
N_CLASSES = 4
RANDOM_STATE = 42
METRIC = "l2"
MODE = "no_loops"

# 输出目录
EXPERIMENT_DIR = "./experiment_output"
DATA_BASE_DIR = "./experiment_data"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(DATA_BASE_DIR, exist_ok=True)


def run_single_experiment(cluster_std):
    """对单个 CLUSTER_STD 值运行完整实验"""
    print(f"\n{'='*60}")
    print(f"实验：CLUSTER_STD = {cluster_std}")
    print(f"{'='*60}")

    # 1. 生成数据集
    data_dir = os.path.join(DATA_BASE_DIR, f"std_{cluster_std}")
    print(f"生成数据集 -> {data_dir}")
    generate_and_save(
        data_dir=data_dir,
        n_samples=N_SAMPLES,
        n_classes=N_CLASSES,
        cluster_std=cluster_std,
        random_state=RANDOM_STATE,
        force=True
    )

    # 2. 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_prepared_dataset(data_dir)

    # 3. 网格搜索最优 k
    print(f"进行 k 值网格搜索：{KS}")
    best_k, val_accs = select_k_by_validation(
        X_train, y_train, X_val, y_val, KS, METRIC, MODE
    )

    # 4. 在测试集上评估 k=1 和 best_k
    X_trv = np.vstack([X_train, X_val])
    y_trv = np.hstack([y_train, y_val])

    # k=1 的性能
    y_pred_k1 = knn_predict(X_test, X_trv, y_trv, k=1, metric=METRIC, mode=MODE)
    acc_k1 = np.mean(y_pred_k1 == y_test)

    # best_k 的性能
    y_pred_best = knn_predict(X_test, X_trv, y_trv, k=best_k, metric=METRIC, mode=MODE)
    acc_best = np.mean(y_pred_best == y_test)

    print(f"验证集最优 k = {best_k}, 验证准确率 = {max(val_accs):.4f}")
    print(f"测试集 k=1 准确率 = {acc_k1:.4f}")
    print(f"测试集 k={best_k} 准确率 = {acc_best:.4f}")

    # 5. 记录结果
    result = {
        'cluster_std': cluster_std,
        'best_k': int(best_k),
        'ks': KS,
        'val_accs': [float(a) for a in val_accs],
        'max_val_acc': float(max(val_accs)),
        'test_acc_k1': float(acc_k1),
        'test_acc_best': float(acc_best),
    }

    return result, X_trv, y_trv, X_test, y_test


def plot_k_vs_std_relationship(results):
    """绘制 CLUSTER_STD 与 best_k 的关系图"""
    stds = [r['cluster_std'] for r in results]
    best_ks = [r['best_k'] for r in results]
    max_accs = [r['max_val_acc'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：CLUSTER_STD vs best_k
    ax1.plot(stds, best_ks, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('CLUSTER_STD (类内标准差)', fontsize=12)
    ax1.set_ylabel('Best K', fontsize=12)
    ax1.set_title('类内方差对最优K值的影响', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 在点上标注 k 值
    for std, k in zip(stds, best_ks):
        ax1.annotate(f'k={k}', (std, k), textcoords="offset points",
                    xytext=(0,8), ha='center', fontsize=9)

    # 子图2：CLUSTER_STD vs 最大验证准确率
    ax2.plot(stds, max_accs, 's-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('CLUSTER_STD (类内标准差)', fontsize=12)
    ax2.set_ylabel('Max Validation Accuracy', fontsize=12)
    ax2.set_title('类内方差对分类准确率的影响', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # 在点上标注准确率
    for std, acc in zip(stds, max_accs):
        ax2.annotate(f'{acc:.3f}', (std, acc), textcoords="offset points",
                    xytext=(0,8), ha='center', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(EXPERIMENT_DIR, 'std_vs_k_relationship.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[图表] 保存 -> {out_path}")


def plot_all_k_curves(results):
    """绘制所有 CLUSTER_STD 的 k-accuracy 曲线"""
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, result in enumerate(results):
        ax = axes[idx]
        std = result['cluster_std']
        ks = result['ks']
        accs = result['val_accs']
        best_k = result['best_k']

        ax.plot(ks, accs, 'o-', linewidth=2, markersize=6)
        ax.axvline(best_k, color='red', linestyle='--', alpha=0.7, label=f'best k={best_k}')
        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('Validation Accuracy', fontsize=10)
        ax.set_title(f'CLUSTER_STD = {std}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xticks(ks)

    # 隐藏多余的子图
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    out_path = os.path.join(EXPERIMENT_DIR, 'all_k_curves.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[图表] 保存 -> {out_path}")


def plot_decision_boundaries_comparison(cluster_std, X_trv, y_trv, X_test, y_test, k1_k, best_k):
    """对比 k=1 和 best_k 的决策边界"""

    # 设置绘图参数
    margin = 0.5
    x_min = min(X_trv[:,0].min(), X_test[:,0].min()) - margin
    x_max = max(X_trv[:,0].max(), X_test[:,0].max()) + margin
    y_min = min(X_trv[:,1].min(), X_test[:,1].min()) - margin
    y_max = max(X_trv[:,1].max(), X_test[:,1].max()) + margin

    grid_n = 200
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_n),
                         np.linspace(y_min, y_max, grid_n))
    grid = np.c_[xx.ravel(), yy.ravel()]

    n_classes = int(max(y_trv.max(), y_test.max())) + 1
    cmap = ListedColormap(plt.cm.tab10.colors[:n_classes])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, k in zip(axes, [k1_k, best_k]):
        # 预测网格点
        Z = knn_predict(grid, X_trv, y_trv, k=k, metric=METRIC, mode=MODE)
        Z = Z.reshape(xx.shape)

        # 绘制背景
        ax.contourf(xx, yy, Z, levels=np.arange(-0.5, n_classes+0.5, 1),
                   cmap=cmap, alpha=0.4)

        # 绘制训练点
        ax.scatter(X_trv[:,0], X_trv[:,1], c=y_trv, cmap=cmap,
                  s=15, edgecolors='k', linewidths=0.3, alpha=0.6, label='Train+Val')

        # 绘制测试点
        y_pred = knn_predict(X_test, X_trv, y_trv, k=k, metric=METRIC, mode=MODE)
        correct = (y_pred == y_test)

        # 正确分类的测试点
        ax.scatter(X_test[correct,0], X_test[correct,1], c=y_test[correct],
                  cmap=cmap, s=60, marker='^', edgecolors='k', linewidths=0.5,
                  label='Test (Correct)')

        # 误分类的测试点
        if np.any(~correct):
            ax.scatter(X_test[~correct,0], X_test[~correct,1],
                      marker='x', s=80, c='red', linewidths=2, label='Misclassified')

        acc = np.mean(correct)
        ax.set_title(f'k = {k} (Test Acc = {acc:.4f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle(f'决策边界对比 (CLUSTER_STD = {cluster_std})',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(EXPERIMENT_DIR, f'boundary_std_{cluster_std}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[图表] 保存 -> {out_path}")


def main():
    """主实验流程"""
    print("="*80)
    print("实验：探索类内方差（CLUSTER_STD）对 kNN 最优 k 值的影响")
    print("="*80)
    print(f"CLUSTER_STD 范围: {CLUSTER_STDS}")
    print(f"k 候选值: {KS}")
    print(f"样本数: {N_SAMPLES}, 类别数: {N_CLASSES}, 随机种子: {RANDOM_STATE}")
    print(f"距离度量: {METRIC}, 模式: {MODE}")
    print("="*80)

    results = []
    datasets = {}

    # 1. 运行所有实验
    total = len(CLUSTER_STDS)
    for idx, cluster_std in enumerate(CLUSTER_STDS, 1):
        print(f"\n[进度] {idx}/{total}")
        result, X_trv, y_trv, X_test, y_test = run_single_experiment(cluster_std)
        results.append(result)
        datasets[cluster_std] = (X_trv, y_trv, X_test, y_test)

    # 2. 保存实验结果
    result_file = os.path.join(EXPERIMENT_DIR, 'experiment_results.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[结果] 保存 -> {result_file}")

    # 3. 生成汇总图表
    print("\n" + "="*80)
    print("生成可视化图表")
    print("="*80)

    plot_k_vs_std_relationship(results)
    plot_all_k_curves(results)

    # 4. 为几个代表性的 CLUSTER_STD 生成决策边界对比图
    representative_stds = [0.5, 2.0, 4.0, 8.0]
    for std in representative_stds:
        if std in datasets:
            result = next(r for r in results if r['cluster_std'] == std)
            X_trv, y_trv, X_test, y_test = datasets[std]
            plot_decision_boundaries_comparison(
                std, X_trv, y_trv, X_test, y_test,
                k1_k=1, best_k=result['best_k']
            )

    # 5. 打印实验总结
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    print(f"{'CLUSTER_STD':<15} {'Best K':<10} {'Val Acc':<12} {'Test Acc (k=1)':<18} {'Test Acc (best_k)':<20}")
    print("-"*80)
    for r in results:
        print(f"{r['cluster_std']:<15.1f} {r['best_k']:<10} {r['max_val_acc']:<12.4f} "
              f"{r['test_acc_k1']:<18.4f} {r['test_acc_best']:<20.4f}")
    print("="*80)

    print(f"\n所有结果已保存到: {EXPERIMENT_DIR}/")
    print("实验完成!")


if __name__ == "__main__":
    main()
