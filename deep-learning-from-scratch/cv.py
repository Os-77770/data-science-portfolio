import numpy as np
from nn_core import (
    initialize_weights,
    gradient_descent,
    predict,
    misclassification_error,
)


def k_fold_split(X, y, k=3):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    folds = np.array_split(indices, k)

    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])

        splits.append((train_idx, val_idx))

    return splits


def cross_validate_perceptron(X, y, learning_rates, lambdas, k=3, iterations=200):
    results = []

    splits = k_fold_split(X, y, k)

    for lr in learning_rates:
        for lam in lambdas:
            train_errors = []
            val_errors = []

            print(f"\nTesting lr={lr}, lambda={lam}")

            for train_idx, val_idx in splits:
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                weights = initialize_weights([784, 10])

                trained_weights = gradient_descent(
                    X_train, y_train, weights,
                    lr=lr,
                    lambda_=lam,
                    iterations=iterations
                )

                train_pred = predict(X_train, trained_weights)
                val_pred = predict(X_val, trained_weights)

                train_err = misclassification_error(y_train, train_pred)
                val_err = misclassification_error(y_val, val_pred)

                train_errors.append(train_err)
                val_errors.append(val_err)

            avg_train = np.mean(train_errors)
            avg_val = np.mean(val_errors)

            print(f"Train Error: {avg_train:.4f}, Val Error: {avg_val:.4f}")

            results.append({
                "lr": lr,
                "lambda": lam,
                "train_error": avg_train,
                "val_error": avg_val
            })

    return results