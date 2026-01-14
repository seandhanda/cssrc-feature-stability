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

    rng = np.random.default_rng(seed)

    for run in range(n_runs):
        split_seed = int(rng.integers(0, 1_000_000_000))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed, stratify=y
        )

        # Logistic Regression
        logreg.fit(X_train, y_train)
        yhat_lr = logreg.predict(X_test)
        acc_lr = accuracy_score(y_test, yhat_lr)

        # feature importance = absolute coefficients
        coef = logreg.named_steps["clf"].coef_.ravel()
        imp_lr = np.abs(coef)
        logreg_importances.append(imp_lr)

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

        rows.append({
            "run": run,
            "split_seed": split_seed,
            "acc_logreg": acc_lr,
            "acc_rf": acc_rf,
        })

    # Convert to arrays
    logreg_importances = np.vstack(logreg_importances)  # (n_runs, n_features)
    rf_importances = np.vstack(rf_importances)

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

    stability_lr = mean_pairwise_spearman(logreg_importances)
    stability_rf = mean_pairwise_spearman(rf_importances)

    var_lr = mean_feature_variance(logreg_importances)
    var_rf = mean_feature_variance(rf_importances)

    summary = pd.DataFrame([{
        "n_runs": n_runs,
        "test_size": test_size,
        "mean_acc_logreg": float(np.mean([r["acc_logreg"] for r in rows])),
        "std_acc_logreg": float(np.std([r["acc_logreg"] for r in rows])),
        "mean_acc_rf": float(np.mean([r["acc_rf"] for r in rows])),
        "std_acc_rf": float(np.std([r["acc_rf"] for r in rows])),
        "stability_spearman_logreg": stability_lr,
        "stability_spearman_rf": stability_rf,
        "mean_feature_var_logreg": var_lr,
        "mean_feature_var_rf": var_rf,
    }])

    runs_df = pd.DataFrame(rows)

    # Save outputs
    runs_path = os.path.join(out_dir, "runs.csv")
    summary_path = os.path.join(out_dir, "summary.csv")
    lr_imps_path = os.path.join(out_dir, "logreg_importances.csv")
    rf_imps_path = os.path.join(out_dir, "rf_importances.csv")

    runs_df.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(logreg_importances, columns=feature_names).to_csv(lr_imps_path, index=False)
    pd.DataFrame(rf_importances, columns=feature_names).to_csv(rf_imps_path, index=False)

    print("Saved:")
    print(" -", runs_path)
    print(" -", summary_path)
    print(" -", lr_imps_path)
    print(" -", rf_imps_path)
    print("\nSummary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main(out_dir="outputs", n_runs=30, test_size=0.30, seed=42)
