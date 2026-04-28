import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)


def plot_param_curve(df, param_col, out_name, title):
    grouped = df.groupby(param_col)[["train_error", "val_error"]].mean().reset_index()
    grouped = grouped.sort_values(param_col)

    plt.figure(figsize=(7, 5))
    plt.plot(grouped[param_col], grouped["train_error"], marker="o", label="CV Train Error")
    plt.plot(grouped[param_col], grouped["val_error"], marker="o", label="CV Validation Error")
    plt.xlabel(param_col)
    plt.ylabel("Misclassification Error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / out_name)
    plt.show()


def main():
    perc = pd.read_csv("../results/cv_results.csv")
    reg = pd.read_csv("../results/regular_dnn_cv_results.csv")
    custom = pd.read_csv("../results/custom_dnn_cv_results.csv")

    
    plot_param_curve(
        perc, "lr",
        "cv_perceptron_lr.png",
        "Perceptron CV Error vs Learning Rate"
    )
    plot_param_curve(
        perc, "lambda",
        "cv_perceptron_lambda.png",
        "Perceptron CV Error vs Lambda"
    )

    
    plot_param_curve(
        reg, "H",
        "cv_regular_H.png",
        "Regular DNN CV Error vs Depth (H)"
    )
    plot_param_curve(
        reg, "B",
        "cv_regular_B.png",
        "Regular DNN CV Error vs Width (B)"
    )
    plot_param_curve(
        reg, "lr",
        "cv_regular_lr.png",
        "Regular DNN CV Error vs Learning Rate"
    )
    plot_param_curve(
        reg, "lambda",
        "cv_regular_lambda.png",
        "Regular DNN CV Error vs Lambda"
    )

    
    plot_param_curve(
        custom, "n1",
        "cv_custom_n1.png",
        "Custom DNN CV Error vs n1"
    )
    plot_param_curve(
        custom, "n2",
        "cv_custom_n2.png",
        "Custom DNN CV Error vs n2"
    )
    plot_param_curve(
        custom, "lr",
        "cv_custom_lr.png",
        "Custom DNN CV Error vs Learning Rate"
    )
    plot_param_curve(
        custom, "lambda",
        "cv_custom_lambda.png",
        "Custom DNN CV Error vs Lambda"
    )


if __name__ == "__main__":
    main()