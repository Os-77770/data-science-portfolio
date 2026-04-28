import numpy as np
import pandas as pd
from pathlib import Path
from nn_core import initialize_weights, gradient_descent, predict


RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data(path):
    data = np.loadtxt(path)
    X = data[:, :784]
    y = data[:, 784].astype(int)
    return X, y


def main():
    X_train, y_train = load_data("../data/optdigits_train.dat")
    X_trial, y_trial = load_data("../data/optdigits_trial.dat")

    # Final chosen best model
    layers = [784, 16, 10]
    lr = 0.05
    lambda_ = 0.0
    iterations = 300

    weights = initialize_weights(layers)
    best_weights = gradient_descent(
        X_train, y_train,
        weights,
        lr=lr,
        lambda_=lambda_,
        iterations=iterations,
        verbose=False
    )

    preds = predict(X_trial, best_weights)

    df = pd.DataFrame({
        "trial_example": list(range(1, len(y_trial) + 1)),
        "true_label": y_trial,
        "predicted_label": preds,
        "correct": preds == y_trial
    })

    print(df)
    df.to_csv(RESULTS_DIR / "trial_predictions.csv", index=False)
    print("\nSaved to ../results/trial_predictions.csv")


if __name__ == "__main__":
    main()