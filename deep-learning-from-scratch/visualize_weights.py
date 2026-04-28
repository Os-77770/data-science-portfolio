import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nn_core import initialize_weights, gradient_descent


RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data(path):
    data = np.loadtxt(path)
    X = data[:, :784]
    y = data[:, 784].astype(int)
    return X, y


def train_model(X, y, layers, lr, lambda_, iterations):
    weights = initialize_weights(layers)
    best_weights = gradient_descent(
        X, y,
        weights,
        lr=lr,
        lambda_=lambda_,
        iterations=iterations,
        verbose=False
    )
    return best_weights


def visualize_matrix_columns(W, title, out_name, count=10):
    plt.figure(figsize=(10, 5))
    for i in range(count):
        plt.subplot(2, 5, i + 1)
        image = W[:, i].reshape(28, 28)
        plt.imshow(image, cmap="gray")
        plt.title(f"Unit {i}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / out_name)
    plt.show()


def main():
    X, y = load_data("../data/optdigits_train.dat")

    
    perc_weights = train_model(X, y, [784, 10], lr=0.05, lambda_=0.0, iterations=200)
    W_perc, _ = perc_weights[0]
    visualize_matrix_columns(
        W_perc,
        "Weight Visualization (Perceptron Output Units)",
        "weights_perceptron.png",
        count=10
    )

    
    reg_weights = train_model(X, y, [784, 16, 10], lr=0.05, lambda_=0.0, iterations=300)
    W_reg_first, _ = reg_weights[0]
    visualize_matrix_columns(
        W_reg_first,
        "Weight Visualization (Regular DNN First Hidden Layer)",
        "weights_regular_dnn.png",
        count=10
    )

    
    custom_weights = train_model(X, y, [784, 32, 16, 10], lr=0.05, lambda_=0.0, iterations=300)
    W_custom_first, _ = custom_weights[0]
    visualize_matrix_columns(
        W_custom_first,
        "Weight Visualization (Custom DNN First Hidden Layer)",
        "weights_custom_dnn.png",
        count=10
    )


if __name__ == "__main__":
    main()