import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nn_core import (
    initialize_weights,
    gradient_descent,
    forward_prop,
    compute_loss,
    predict,
    misclassification_error,
)


RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data(path):
    data = np.loadtxt(path)
    X = data[:, :784]
    y = data[:, 784].astype(int)
    return X, y


def evaluate_model(X_train_sub, y_train_sub, X_test, y_test, layers, lr, lambda_, iterations):
    weights = initialize_weights(layers)
    weights = gradient_descent(
        X_train_sub,
        y_train_sub,
        weights,
        lr=lr,
        lambda_=lambda_,
        iterations=iterations,
        verbose=False,
    )

    train_probs, train_cache = forward_prop(X_train_sub, weights)
    test_probs, test_cache = forward_prop(X_test, weights)

    train_loss = compute_loss(y_train_sub, weights, train_cache, lambda_)
    test_loss = compute_loss(y_test, weights, test_cache, lambda_)

    train_pred = predict(X_train_sub, weights)
    test_pred = predict(X_test, weights)

    train_err = misclassification_error(y_train_sub, train_pred)
    test_err = misclassification_error(y_test, test_pred)

    return train_loss, test_loss, train_err, test_err


def plot_curve(sizes, train_values, test_values, ylabel, title, out_name):
    plt.figure(figsize=(7, 5))
    plt.plot(sizes, train_values, marker="o", label="Train")
    plt.plot(sizes, test_values, marker="o", label="Test")
    plt.xlabel("Training Size (m)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / out_name)
    plt.show()


def run_learning_curve(model_name, layers, lr, lambda_, iterations):
    X_train, y_train = load_data("../data/optdigits_train.dat")
    X_test, y_test = load_data("../data/optdigits_test.dat")

    sizes = [10, 40, 100, 200, 400, 800, 1600]

    train_losses, test_losses = [], []
    train_errors, test_errors = [], []

    for m in sizes:
        print(f"{model_name}: training with m={m}")
        X_sub = X_train[:m]
        y_sub = y_train[:m]

        train_loss, test_loss, train_err, test_err = evaluate_model(
            X_sub, y_sub, X_test, y_test, layers, lr, lambda_, iterations
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_errors.append(train_err)
        test_errors.append(test_err)

    safe_name = model_name.lower().replace(" ", "_")

    plot_curve(
        sizes, train_losses, test_losses,
        "Proxy Loss",
        f"{model_name} Learning Curve (Proxy Loss)",
        f"{safe_name}_learning_proxy.png"
    )

    plot_curve(
        sizes, train_errors, test_errors,
        "Misclassification Error",
        f"{model_name} Learning Curve (Misclassification Error)",
        f"{safe_name}_learning_error.png"
    )


def main():
    run_learning_curve(
        model_name="Perceptron",
        layers=[784, 10],
        lr=0.05,
        lambda_=0.0,
        iterations=200,
    )

    run_learning_curve(
        model_name="Regular DNN",
        layers=[784, 16, 10],
        lr=0.05,
        lambda_=0.0,
        iterations=300,
    )

    run_learning_curve(
        model_name="Custom DNN",
        layers=[784, 32, 16, 10],
        lr=0.05,
        lambda_=0.0,
        iterations=300,
    )


if __name__ == "__main__":
    main()