import numpy as np
from nn_core import (
    initialize_weights,
    gradient_descent,
    predict,
    misclassification_error,
)
from runtime_utils import adaptive_iterations


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



def cross_validate_regular_dnn(X, y, depths, widths, learning_rates, lambdas, k=3):
    splits = k_fold_split(X, y, k)
    results = []

    for H in depths:
        for B in widths:
            layers = [784] + [B] * H + [10]

            for lr in learning_rates:
                for lam in lambdas:
                    train_errors = []
                    val_errors = []

                    print(f"\nRegular DNN: H={H}, B={B}, lr={lr}, lambda={lam}")

                    for train_idx, val_idx in splits:
                        X_train, y_train = X[train_idx], y[train_idx]
                        X_val, y_val = X[val_idx], y[val_idx]

                        weights = initialize_weights(layers)

                        R = adaptive_iterations(
                            reference_layers=[784, 64, 64, 10],
                            reference_m=1000,
                            reference_R=1000,
                            target_layers=layers,
                            target_m=len(X_train)
                        )

                        weights = gradient_descent(
                            X_train, y_train,
                            weights,
                            lr=lr,
                            lambda_=lam,
                            iterations=R,
                            verbose=False
                        )

                        train_pred = predict(X_train, weights)
                        val_pred = predict(X_val, weights)

                        train_err = misclassification_error(y_train, train_pred)
                        val_err = misclassification_error(y_val, val_pred)

                        train_errors.append(train_err)
                        val_errors.append(val_err)

                    results.append({
                        "type": "regular",
                        "H": H,
                        "B": B,
                        "lr": lr,
                        "lambda": lam,
                        "iterations": R,
                        "train_error": np.mean(train_errors),
                        "val_error": np.mean(val_errors)
                    })

    return results



def cross_validate_custom_dnn(X, y, n1_list, n2_list, learning_rates, lambdas, k=3):
    splits = k_fold_split(X, y, k)
    results = []

    for n1 in n1_list:
        for n2 in n2_list:
            if n2 >= n1:
                continue

            layers = [784, n1, n2, 10]

            for lr in learning_rates:
                for lam in lambdas:
                    train_errors = []
                    val_errors = []

                    print(f"\nCustom DNN: n1={n1}, n2={n2}, lr={lr}, lambda={lam}")

                    for train_idx, val_idx in splits:
                        X_train, y_train = X[train_idx], y[train_idx]
                        X_val, y_val = X[val_idx], y[val_idx]

                        weights = initialize_weights(layers)

                        R = adaptive_iterations(
                            reference_layers=[784, 64, 64, 10],
                            reference_m=1000,
                            reference_R=1000,
                            target_layers=layers,
                            target_m=len(X_train)
                        )

                        weights = gradient_descent(
                            X_train, y_train,
                            weights,
                            lr=lr,
                            lambda_=lam,
                            iterations=R,
                            verbose=False
                        )

                        train_pred = predict(X_train, weights)
                        val_pred = predict(X_val, weights)

                        train_err = misclassification_error(y_train, train_pred)
                        val_err = misclassification_error(y_val, val_pred)

                        train_errors.append(train_err)
                        val_errors.append(val_err)

                    results.append({
                        "type": "custom",
                        "n1": n1,
                        "n2": n2,
                        "lr": lr,
                        "lambda": lam,
                        "iterations": R,
                        "train_error": np.mean(train_errors),
                        "val_error": np.mean(val_errors)
                    })

    return results