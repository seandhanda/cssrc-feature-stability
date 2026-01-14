import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main(out_dir="outputs", fig_dir="figures"):
    os.makedirs(fig_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(out_dir, "summary.csv"))
    runs = pd.read_csv(os.path.join(out_dir, "runs.csv"))
    lr_imps = pd.read_csv(os.path.join(out_dir, "logreg_importances.csv"))
    rf_imps = pd.read_csv(os.path.join(out_dir, "rf_importances.csv"))

    # 1) Accuracy Across Resamples — Boxplot (Legend Outside)
    plt.figure(figsize=(6.5, 4))
    plt.boxplot(
        [runs["acc_logreg"], runs["acc_rf"]],
        labels=["Logistic Regression", "Random Forest"],
        showmeans=True
    )
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across 30 Resampled Train/Test Splits")

    legend_elements = [
        Line2D([0], [0], marker="^", color="green", linestyle="None",
               markersize=7, label="Mean"),
        Line2D([0], [0], color="orange", linewidth=2, label="Median"),
    ]

    plt.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=9
    )

    p1 = os.path.join(fig_dir, "fig_accuracy_box.png")
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Feature Importance Variability
    # |coef| = absolute value of logistic regression coefficients
    lr_var = lr_imps.abs().var(axis=0).sort_values(ascending=False)
    rf_var = rf_imps.var(axis=0).sort_values(ascending=False)

    topk = 10
    plt.figure(figsize=(9, 4))
    x = np.arange(topk)
    plt.bar(
        x - 0.2,
        lr_var.head(topk).values,
        width=0.4,
        label="Logistic Regression |Coef| Variance"
    )
    plt.bar(
        x + 0.2,
        rf_var.head(topk).values,
        width=0.4,
        label="Random Forest Importance Variance"
    )
    plt.xticks(x, lr_var.head(topk).index, rotation=45, ha="right")
    plt.ylabel("Variance Across Resamples")
    plt.title("Top-10 Most Variable Features Across Resamples")
    plt.legend(fontsize=9)

    p2 = os.path.join(fig_dir, "fig_top_feature_variance.png")
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()

    # 3) Performance and Stability Summary Table
    plt.figure(figsize=(10.5, 2.6))
    plt.axis("off")
    s = summary.iloc[0]

    table_data = [
        ["Model", "Mean Accuracy (± Std)", "Stability (Mean Spearman)", "Mean Feature Variance"],
        [
            "Logistic Regression",
            f"{s['mean_acc_logreg']:.3f} ± {s['std_acc_logreg']:.3f}",
            f"{s['stability_spearman_logreg']:.3f}",
            f"{s['mean_feature_var_logreg']:.6f}",
        ],
        [
            "Random Forest",
            f"{s['mean_acc_rf']:.3f} ± {s['std_acc_rf']:.3f}",
            f"{s['stability_spearman_rf']:.3f}",
            f"{s['mean_feature_var_rf']:.6f}",
        ],
    ]

    table = plt.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.22, 0.30, 0.28, 0.20]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    plt.title("Performance and Stability Summary", pad=10)

    p3 = os.path.join(fig_dir, "fig_summary_table.png")
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved Figures:")
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)

if __name__ == "__main__":
    main()
