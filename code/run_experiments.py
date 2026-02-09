import os
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def rankdata_desc(x: np.ndarray) -> np.ndarray:
    """
    Convert values to ranks (1 = most important). Descending by absolute value.
    Simple stable tie-breaking via argsort; good enough for this study.
    """
    order = np.argsort(-np.abs(x), kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks

def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation via Pearson correlation of ranks."""
    ra = rankdata_desc(a)
    rb = rankdata_desc(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

def corr_prune_mask_spearman(X_train: np.ndarray, threshold: float = 0.90) -> np.ndarray:
    """
    Greedy correlation filter using absolute Spearman correlation on X_train only.
    Returns a boolean mask over original features: True = keep, False = drop.

    Deterministic rule: if two kept features exceed threshold, drop the later-index feature.
    """
    df = pd.DataFrame(X_train)
    C = df.corr(method="spearman").abs().to_numpy()
    n = C.shape[0]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if keep[j] and C[i, j] > threshold:
                keep[j] = False
    return keep


def main(
    out_dir: str = "outputs",
    n_runs: int = 30,
    test_size: float = 0.30,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    # Dataset: Breast Cancer Wisconsin (Diagnostic)
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)

    # Models
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, C=1.0, penalty="l2", solver="lbfgs"))
    ])

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=seed,
        n_jobs=-1
    )

    # Storage
    rows = []
    logreg_importances = []
    rf_importances = []

    logreg_importances_pruned = []
    rf_importances_pruned = []


    rng = np.random.default_rng(seed)

    for run in range(n_runs):
        split_seed = int(rng.integers(0, 1_000_000_000))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed, stratify=y
        )

        # Correlation pruning mask computed on TRAINING SPLIT ONLY (no leakage)
        mask_keep = corr_prune_mask_spearman(X_train, threshold=0.90)
        X_train_p = X_train[:, mask_keep]
        X_test_p = X_test[:, mask_keep]
        n_kept = int(mask_keep.sum())

        # Logistic Regression
        logreg.fit(X_train, y_train)
        yhat_lr = logreg.predict(X_test)
        acc_lr = accuracy_score(y_test, yhat_lr)

        # feature importance = absolute coefficients
        coef = logreg.named_steps["clf"].coef_.ravel()
        imp_lr = np.abs(coef)
        logreg_importances.append(imp_lr)

        # Logistic Regression (PRUNED)
        logreg.fit(X_train_p, y_train)
        yhat_lr_p = logreg.predict(X_test_p)
        acc_lr_p = accuracy_score(y_test, yhat_lr_p)

        coef_p = logreg.named_steps["clf"].coef_.ravel()
        imp_lr_p_reduced = np.abs(coef_p)

        # Expand back to full length (zeros for removed features) so stability is comparable
        imp_lr_p_full = np.zeros(len(feature_names), dtype=float)
        imp_lr_p_full[mask_keep] = imp_lr_p_reduced
        logreg_importances_pruned.append(imp_lr_p_full)


        # Random Forest
        rf_run = RandomForestClassifier(
            n_estimators=400,
            random_state=split_seed,
            n_jobs=-1
        )
        rf_run.fit(X_train, y_train)
        yhat_rf = rf_run.predict(X_test)
        acc_rf = accuracy_score(y_test, yhat_rf)

        imp_rf = rf_run.feature_importances_
        rf_importances.append(imp_rf)

        # Random Forest (PRUNED)
        rf_run_p = RandomForestClassifier(
            n_estimators=400,
            random_state=split_seed,
            n_jobs=-1
        )
        rf_run_p.fit(X_train_p, y_train)
        yhat_rf_p = rf_run_p.predict(X_test_p)
        acc_rf_p = accuracy_score(y_test, yhat_rf_p)

        imp_rf_p_reduced = rf_run_p.feature_importances_

        imp_rf_p_full = np.zeros(len(feature_names), dtype=float)
        imp_rf_p_full[mask_keep] = imp_rf_p_reduced
        rf_importances_pruned.append(imp_rf_p_full)

        rows.append({
            "run": run,
            "split_seed": split_seed,
            "acc_logreg": acc_lr,
            "acc_rf": acc_rf,
            "acc_logreg_pruned": acc_lr_p,
            "acc_rf_pruned": acc_rf_p,
            "n_features_kept": n_kept,
        })


    # Convert to arrays
    logreg_importances = np.vstack(logreg_importances)  # (n_runs, n_features)
    rf_importances = np.vstack(rf_importances)

    logreg_importances_pruned = np.vstack(logreg_importances_pruned)
    rf_importances_pruned = np.vstack(rf_importances_pruned)

    def pairwise_spearman_corrs(imps: np.ndarray) -> np.ndarray:
        corrs = []
        for i in range(imps.shape[0]):
            for j in range(i + 1, imps.shape[0]):
                corrs.append(spearman_corr(imps[i], imps[j]))
        return np.asarray(corrs, dtype=float)

    def bootstrap_mean_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int = 2000, alpha: float = 0.05):
        """
        Bootstrap CI for the mean of x (resample with replacement).
        Returns (mean, ci_low, ci_high).
        """
        x = x[~np.isnan(x)]
        mean = float(np.mean(x))
        n = len(x)
        boot_means = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_means[b] = np.mean(x[idx])
        lo = float(np.quantile(boot_means, alpha / 2))
        hi = float(np.quantile(boot_means, 1 - alpha / 2))
        return mean, lo, hi


    # Stability metrics:
    # 1) Mean pairwise Spearman correlation of feature-importance vectors across runs
    def mean_pairwise_spearman(imps: np.ndarray) -> float:
        corrs = []
        for i in range(imps.shape[0]):
            for j in range(i + 1, imps.shape[0]):
                corrs.append(spearman_corr(imps[i], imps[j]))
        return float(np.nanmean(corrs))

    # 2) Average coefficient/importance variance across features
    def mean_feature_variance(imps: np.ndarray) -> float:
        return float(np.mean(np.var(imps, axis=0)))

    var_lr = mean_feature_variance(logreg_importances)
    var_rf = mean_feature_variance(rf_importances)

    # Pairwise correlations (for bootstrap CI)
    ci_rng = np.random.default_rng(seed + 12345)

    lr_corrs = pairwise_spearman_corrs(logreg_importances)
    rf_corrs = pairwise_spearman_corrs(rf_importances)
    lrp_corrs = pairwise_spearman_corrs(logreg_importances_pruned)
    rfp_corrs = pairwise_spearman_corrs(rf_importances_pruned)

    stability_lr, stability_lr_ci_low, stability_lr_ci_high = bootstrap_mean_ci(lr_corrs, ci_rng)
    stability_rf, stability_rf_ci_low, stability_rf_ci_high = bootstrap_mean_ci(rf_corrs, ci_rng)
    stability_lr_p, stability_lr_p_ci_low, stability_lr_p_ci_high = bootstrap_mean_ci(lrp_corrs, ci_rng)
    stability_rf_p, stability_rf_p_ci_low, stability_rf_p_ci_high = bootstrap_mean_ci(rfp_corrs, ci_rng)


    var_lr_p = mean_feature_variance(logreg_importances_pruned)
    var_rf_p = mean_feature_variance(rf_importances_pruned)


    summary = pd.DataFrame([{
        "n_runs": n_runs,
        "test_size": test_size,
        "mean_acc_logreg": float(np.mean([r["acc_logreg"] for r in rows])),
        "mean_acc_rf": float(np.mean([r["acc_rf"] for r in rows])),
        "stability_spearman_logreg": stability_lr,
        "stability_spearman_rf": stability_rf,
        "mean_feature_var_logreg": var_lr,
        "mean_feature_var_rf": var_rf,
        "mean_acc_logreg_pruned": float(np.mean([r["acc_logreg_pruned"] for r in rows])),
        "mean_acc_rf_pruned": float(np.mean([r["acc_rf_pruned"] for r in rows])),
        "std_acc_logreg": float(np.std([r["acc_logreg"] for r in rows], ddof=1)),
        "std_acc_rf": float(np.std([r["acc_rf"] for r in rows], ddof=1)),
        "std_acc_logreg_pruned": float(np.std([r["acc_logreg_pruned"] for r in rows], ddof=1)),
        "std_acc_rf_pruned": float(np.std([r["acc_rf_pruned"] for r in rows], ddof=1)),

        "stability_spearman_logreg_pruned": stability_lr_p,
        "stability_spearman_rf_pruned": stability_rf_p,
        "mean_feature_var_logreg_pruned": var_lr_p,
        "mean_feature_var_rf_pruned": var_rf_p,

        "mean_n_features_kept": float(np.mean([r["n_features_kept"] for r in rows])),
        "stability_spearman_logreg_ci_low": stability_lr_ci_low,
        "stability_spearman_logreg_ci_high": stability_lr_ci_high,
        "stability_spearman_rf_ci_low": stability_rf_ci_low,
        "stability_spearman_rf_ci_high": stability_rf_ci_high,

        "stability_spearman_logreg_pruned_ci_low": stability_lr_p_ci_low,
        "stability_spearman_logreg_pruned_ci_high": stability_lr_p_ci_high,
        "stability_spearman_rf_pruned_ci_low": stability_rf_p_ci_low,
        "stability_spearman_rf_pruned_ci_high": stability_rf_p_ci_high,
    }])

    runs_df = pd.DataFrame(rows)

    # Save outputs
    runs_path = os.path.join(out_dir, "runs.csv")
    summary_path = os.path.join(out_dir, "summary.csv")
    lr_imps_path = os.path.join(out_dir, "logreg_importances.csv")
    rf_imps_path = os.path.join(out_dir, "rf_importances.csv")
    lr_imps_pruned_path = os.path.join(out_dir, "logreg_importances_pruned.csv")
    rf_imps_pruned_path = os.path.join(out_dir, "rf_importances_pruned.csv")

    runs_df.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(logreg_importances, columns=feature_names).to_csv(lr_imps_path, index=False)
    pd.DataFrame(rf_importances, columns=feature_names).to_csv(rf_imps_path, index=False)
    pd.DataFrame(logreg_importances_pruned, columns=feature_names).to_csv(lr_imps_pruned_path, index=False)
    pd.DataFrame(rf_importances_pruned, columns=feature_names).to_csv(rf_imps_pruned_path, index=False)


    print("Saved:")
    print(" -", runs_path)
    print(" -", summary_path)
    print(" -", lr_imps_path)
    print(" -", rf_imps_path)
    print(" -", lr_imps_pruned_path)
    print(" -", rf_imps_pruned_path)
    print("\nSummary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main(out_dir="outputs", n_runs=30, test_size=0.30, seed=42)
