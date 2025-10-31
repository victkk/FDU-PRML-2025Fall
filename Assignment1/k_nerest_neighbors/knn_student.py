import os
import numpy as np
from typing import List, Tuple
from data_generate import load_prepared_dataset
from viz_knn import plot_k_curve, plot_decision_boundary_multi

# 输出目录
OUT_DIR = "./output"
DATA_DIR = "./input_knn"
FIG_K_CURVE   = os.path.join(OUT_DIR, "knn_k_curve.png")
FIG_BOUNDARY  = os.path.join(OUT_DIR, "knn_boundary.png")

# ============ TODO 1：pairwise_dist ============
def pairwise_dist(X_test, X_train, metric, mode):
    """
    Compute pairwise distances between X_test (Nte,D) and X_train (Ntr,D).

    Required:
      - L2 distance 'l2' with modes:
          * 'two_loops'  
          * 'no_loops' 
      - 'cosine' distance (distance = 1 - cosine_similarity)
    """
    X_test = np.asarray(X_test, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)
    Nte, D  = X_test.shape
    Ntr, D2 = X_train.shape
    assert D == D2, "Dim mismatch between test and train."

    if metric == "l2":
        if mode == "two_loops":
            # =============== TODO (students, REQUIRED) ===============
            dists = np.zeros((Nte, Ntr))
            for i in range(Nte):
                for j in range(Ntr):
                    dists[i, j] = np.sqrt(np.sum((X_test[i] - X_train[j]) ** 2))
            return dists
            # =========================================================

        elif mode == "no_loops":
            # =============== TODO (students, REQUIRED) ===============
            # Using the formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)  # (Nte, 1)
            train_sq = np.sum(X_train ** 2, axis=1, keepdims=True).T  # (1, Ntr)
            cross = 2 * np.dot(X_test, X_train.T)  # (Nte, Ntr)
            dists = np.sqrt(test_sq + train_sq - cross)
            return dists
            # =========================================================

        else:
            raise ValueError("Unknown mode for L2.")

    elif metric == "cosine":
        # =============== TODO (students, REQUIRED) ===============
        # Cosine distance = 1 - cosine_similarity
        # cosine_similarity = (a·b) / (||a|| * ||b||)
        # Normalize vectors
        test_norm = np.linalg.norm(X_test, axis=1, keepdims=True)  # (Nte, 1)
        train_norm = np.linalg.norm(X_train, axis=1, keepdims=True)  # (Ntr, 1)

        # Avoid division by zero
        test_norm = np.where(test_norm == 0, 1, test_norm)
        train_norm = np.where(train_norm == 0, 1, train_norm)

        # Compute cosine similarity
        cos_sim = np.dot(X_test / test_norm, (X_train / train_norm).T)
        # Cosine distance
        dists = 1 - cos_sim
        return dists
        # ================================================
    else:
        raise ValueError("metric must be 'l2' or 'cosine'.")


# ============ TODO 2：knn_predict（多数表决） ============
def knn_predict(X_test, X_train, y_train, k, metric, mode):
    """
    kNN prediction.
    Required: majority vote with L2 distance.

    Returns
    -------
    y_pred : (Nte,) int
    """
    dists = pairwise_dist(X_test, X_train, metric=metric, mode=mode)
    y_train = np.asarray(y_train).reshape(-1).astype(int)
    Nte = dists.shape[0]
    y_pred = np.zeros(Nte, dtype=int)

    for i in range(Nte):
        idx = np.argsort(dists[i])[:k]
        neighbors = y_train[idx]

        # =============== TODO (students, REQUIRED) ===============
        # Majority vote: find the most common label
        # In case of tie, return the minimum label
        unique_labels, counts = np.unique(neighbors, return_counts=True)
        max_count = np.max(counts)
        # Find all labels with max count
        candidates = unique_labels[counts == max_count]
        # Return the minimum label in case of tie
        y_pred[i] = np.min(candidates)
        # ===========================================

    return y_pred


# ============ TODO 3：select_k_by_validation ============
def select_k_by_validation(X_train, y_train, X_val, y_val, ks: List[int], metric, mode) -> Tuple[int, List[float]]:
    """
    Grid-search K on validation set.

    Returns
    -------
    best_k : int
    accs   : list of validation accuracies aligned with ks
    """
    # =============== TODO (students, REQUIRED) ===============
    accs = []
    for k in ks:
        # Predict on validation set with current k
        y_pred = knn_predict(X_val, X_train, y_train, k, metric, mode)
        # Calculate accuracy
        acc = np.mean(y_pred == y_val)
        accs.append(acc)

    # Find the k with highest validation accuracy
    best_idx = np.argmax(accs)
    best_k = ks[best_idx]

    return best_k, accs
    # =========================================================


def run_with_visualization():
    X_train, y_train, X_val, y_val, X_test, y_test = load_prepared_dataset(DATA_DIR)

    ks = [1, 3, 5, 7, 9, 11, 13]
    metric = "l2"           # ["l2", "cosine"]
    mode   = "no_loops"     # ["two_loops", "no_loops", "one_loop"]

    best_k, accs = select_k_by_validation(X_train, y_train, X_val, y_val,
                                          ks, metric=metric, mode=mode)
    print(f"[ModelSelect] best k={best_k} (val acc={max(accs):.4f})")
    plot_k_curve(ks, accs, os.path.join(OUT_DIR, "knn_k_curve.png"))

    X_trv = np.vstack([X_train, X_val]); y_trv = np.hstack([y_train, y_val])
    def predict_fn_for_k(k):
        return lambda Xq: knn_predict(Xq, X_trv, y_trv, k, metric=metric, mode=mode)

    ks_panel = sorted(set(ks + [best_k]))
    plot_decision_boundary_multi(predict_fn_for_k, X_train, y_train, X_test, y_test,
                                 ks=ks_panel,
                                 out_path=os.path.join(OUT_DIR, "knn_boundary_grid.png"),
                                 grid_n=200, batch_size=4096)


if __name__ == "__main__":
    run_with_visualization()
