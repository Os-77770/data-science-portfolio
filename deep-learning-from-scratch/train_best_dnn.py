import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nn_core import train_and_record


RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data(path):
    data = np.loadtxt(path)
    X = data[:, :784]
    y = data[:, 784].astype(int)
    return X, y


def main():
    X_train, y_train = load_data("../data/optdigits_train.dat")
    X_test, y_test = load_data("../data/optdigits_test.dat")

    layers = [784, 16, 10]
    lr = 0.05
    lambda_ = 0.0
    iterations = 300

    best_weights, train_losses, test_losses, train_errors, test_errors = train_and_record(
        X_train, y_train,
        X_test, y_test,
        layers,
        lr,
        lambda_,
        iterations
    )

    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.title("Regular DNN Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "regular_dnn_loss_curve.png")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(train_errors, label="Train Error")
    plt.plot(test_errors, label="Test Error")
    plt.title("Regular DNN Error Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "regular_dnn_error_curve.png")
    plt.show()


if __name__ == "__main__":
    main()