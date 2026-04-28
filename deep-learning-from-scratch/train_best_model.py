import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nn_core import (
    initialize_weights,
    forward_prop,
    compute_loss,
    misclassification_error,
    back_prop
)


RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data(path):
    data = np.loadtxt(path)
    X = data[:, :784]
    y = data[:, 784].astype(int)
    return X, y


def train_and_record(X_train, y_train, X_test, y_test, layers, lr, lambda_, iterations):
    weights = initialize_weights(layers)

    train_losses = []
    test_losses = []
    train_errors = []
    test_errors = []

    for i in range(iterations):
        y_pred_train, cache_train = forward_prop(X_train, weights)
        y_pred_test, cache_test = forward_prop(X_test, weights)

        train_loss = compute_loss(y_train, weights, cache_train, lambda_)
        test_loss = compute_loss(y_test, weights, cache_test, lambda_)

        train_pred = np.argmax(y_pred_train, axis=1)
        test_pred = np.argmax(y_pred_test, axis=1)

        train_err = misclassification_error(y_train, train_pred)
        test_err = misclassification_error(y_test, test_pred)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_errors.append(train_err)
        test_errors.append(test_err)

        grads = back_prop(X_train, y_train, weights, cache_train, lambda_)

        new_weights = []
        for (W, b), (dW, db) in zip(weights, grads):
            W = W - lr * dW
            b = b - lr * db
            new_weights.append((W, b))
        weights = new_weights

        if i % 50 == 0:
            print(f"Iteration {i}, Train Error: {train_err:.4f}, Test Error: {test_err:.4f}")

    return train_losses, test_losses, train_errors, test_errors


def main():
    X_train, y_train = load_data("../data/optdigits_train.dat")
    X_test, y_test = load_data("../data/optdigits_test.dat")

    layers = [784, 10]
    lr = 0.05
    lambda_ = 0.0
    iterations = 300

    train_losses, test_losses, train_errors, test_errors = train_and_record(
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
    plt.title("Perceptron Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "perceptron_loss_curve.png")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(train_errors, label="Train Error")
    plt.plot(test_errors, label="Test Error")
    plt.title("Perceptron Error Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "perceptron_error_curve.png")
    plt.show()


if __name__ == "__main__":
    main()