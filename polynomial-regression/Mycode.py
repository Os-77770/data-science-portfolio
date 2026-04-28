from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as la
from dataclasses import dataclass
from typing import Dict, Tuple, List

TRAIN_PATH = "train.dat"
TEST_PATH = "test.dat"
N_FOLDS = 12
FOLD_SIZE = 5
DEG_MIN = 0
DEG_MAX = 20

@dataclass
class StandardScaler1D:
    mean_: float = 0.0
    std_: float = 1.0

    def fit(self, x: np.ndarray) -> "StandardScaler1D":
        x = np.asarray(x).reshape(-1)
        self.mean_ = float(x.mean())
        self.std_ = float(x.std(ddof=1))
        if self.std_ == 0.0:
            self.std_ = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return (x - self.mean_) / self.std_

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        return z * self.std_ + self.mean_

@dataclass
class ColumnScaler:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ColumnScaler":
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=1)
        self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return (X - self.mean_) / self.std_

def make_poly_features(x_scaled: np.ndarray, degree: int) -> np.ndarray:
    x_scaled = np.asarray(x_scaled).reshape(-1)
    
    return np.vstack([x_scaled ** i for i in range(degree + 1)]).T

def fit_ridge(Phi: np.ndarray, y_scaled: np.ndarray, lam: float) -> np.ndarray:
   
    Phi = np.asarray(Phi)
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    d = Phi.shape[1]
    A = Phi.T @ Phi + lam * np.eye(d)
    b = Phi.T @ y_scaled
    w = la.solve(A, b)
    return w.reshape(-1)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def cv_folds(m: int) -> List[np.ndarray]:
    assert m == N_FOLDS * FOLD_SIZE, f"Expected {N_FOLDS*FOLD_SIZE} training examples, got {m}"
    idx = np.arange(m)
    return [idx[i * FOLD_SIZE:(i + 1) * FOLD_SIZE] for i in range(N_FOLDS)]


def train_model(x_train_orig: np.ndarray, y_train_orig: np.ndarray, degree: int, lam: float) -> Dict:
    xsc = StandardScaler1D().fit(x_train_orig)
    ysc = StandardScaler1D().fit(y_train_orig)

    xtr = xsc.transform(x_train_orig)
    ytr = ysc.transform(y_train_orig)

    Phi_tr = make_poly_features(xtr, degree)

    feat_scaler = None
    if degree >= 1:
        feat_scaler = ColumnScaler().fit(Phi_tr[:, 1:])
        Phi_tr_scaled = np.column_stack([Phi_tr[:, 0], feat_scaler.transform(Phi_tr[:, 1:])])
    else:
        Phi_tr_scaled = Phi_tr

    w = fit_ridge(Phi_tr_scaled, ytr, lam)
    return {"degree": degree, "lambda": lam, "w": w, "xsc": xsc, "ysc": ysc, "fsc": feat_scaler}

def predict(model: Dict, x_orig: np.ndarray) -> np.ndarray:
    degree = int(model["degree"])
    xsc: StandardScaler1D = model["xsc"]
    ysc: StandardScaler1D = model["ysc"]
    fsc: ColumnScaler | None = model["fsc"]
    w: np.ndarray = model["w"]

    xs = xsc.transform(x_orig)
    Phi = make_poly_features(xs, degree)

    if degree >= 1:
        Phi_scaled = np.column_stack([Phi[:, 0], fsc.transform(Phi[:, 1:])])
    else:
        Phi_scaled = Phi

    y_scaled_pred = Phi_scaled @ w
    return ysc.inverse_transform(y_scaled_pred)


def run_cv_degrees(train_data: np.ndarray, degrees: range) -> pd.DataFrame:
    x = train_data[:, 0]
    y = train_data[:, 1]
    folds = cv_folds(len(train_data))

    rows = []
    for d in degrees:
        tr_list, te_list = [], []
        for k, te_idx in enumerate(folds, start=1):
            tr_idx = np.setdiff1d(np.arange(len(train_data)), te_idx, assume_unique=True)

            model = train_model(x[tr_idx], y[tr_idx], degree=d, lam=0.0)
            y_tr_pred = predict(model, x[tr_idx])
            y_te_pred = predict(model, x[te_idx])

            rtr = rmse(y[tr_idx], y_tr_pred)
            rte = rmse(y[te_idx], y_te_pred)

            tr_list.append(rtr)
            te_list.append(rte)

            rows.append({"degree": d, "lambda": 0.0, "fold": k, "rmse_train": rtr, "rmse_test": rte})

        rows.append({"degree": d, "lambda": 0.0, "fold": 0,
                     "rmse_train": float(np.mean(tr_list)), "rmse_test": float(np.mean(te_list)),
                     "avg": True})
    return pd.DataFrame(rows)

def lambda_candidates() -> List[float]:
    zs = list(range(-30, 12, 2))  # -30, -28, ..., 10
    return [0.0] + [float(np.exp(z)) for z in zs]

def run_cv_lambdas(train_data: np.ndarray, degree: int, lambdas: List[float]) -> pd.DataFrame:
    x = train_data[:, 0]
    y = train_data[:, 1]
    folds = cv_folds(len(train_data))

    rows = []
    for lam in lambdas:
        tr_list, te_list = [], []
        for k, te_idx in enumerate(folds, start=1):
            tr_idx = np.setdiff1d(np.arange(len(train_data)), te_idx, assume_unique=True)

            model = train_model(x[tr_idx], y[tr_idx], degree=degree, lam=lam)
            y_tr_pred = predict(model, x[tr_idx])
            y_te_pred = predict(model, x[te_idx])

            rtr = rmse(y[tr_idx], y_tr_pred)
            rte = rmse(y[te_idx], y_te_pred)

            tr_list.append(rtr)
            te_list.append(rte)

            rows.append({"degree": degree, "lambda": lam, "fold": k, "rmse_train": rtr, "rmse_test": rte})

        rows.append({"degree": degree, "lambda": lam, "fold": 0,
                     "rmse_train": float(np.mean(tr_list)), "rmse_test": float(np.mean(te_list)),
                     "avg": True})
    return pd.DataFrame(rows)

def main() -> None:
    train_data = np.loadtxt(TRAIN_PATH)
    test_data = np.loadtxt(TEST_PATH)

   
    deg_df = run_cv_degrees(train_data, range(DEG_MIN, DEG_MAX + 1))
    deg_avg = deg_df[deg_df.get("avg", False) == True].copy()
    deg_avg = deg_avg.sort_values("degree")

    d_star = int(deg_avg.loc[deg_avg["rmse_test"].idxmin(), "degree"])
    print(f"d* (min avg CV test RMSE) = {d_star}")

    deg_avg[["degree", "rmse_train", "rmse_test"]].to_csv("cv_degree_averages.csv", index=False)

    
    lams = lambda_candidates()
    lam_df = run_cv_lambdas(train_data, degree=20, lambdas=lams)
    lam_avg = lam_df[lam_df.get("avg", False) == True].copy()
    lam_star = float(lam_avg.loc[lam_avg["rmse_test"].idxmin(), "lambda"])
    print(f"lambda* (min avg CV test RMSE for degree 20) = {lam_star} (log={math.log(lam_star) if lam_star>0 else 'NA'})")

    lam_avg[["lambda", "rmse_train", "rmse_test"]].to_csv("cv_lambda_averages_deg20.csv", index=False)

    
    x_tr, y_tr = train_data[:, 0], train_data[:, 1]
    x_te, y_te = test_data[:, 0], test_data[:, 1]

    model_d = train_model(x_tr, y_tr, degree=d_star, lam=0.0)
    model_l = train_model(x_tr, y_tr, degree=20, lam=lam_star)

    yhat_tr_d = predict(model_d, x_tr)
    yhat_te_d = predict(model_d, x_te)
    yhat_tr_l = predict(model_l, x_tr)
    yhat_te_l = predict(model_l, x_te)

    print("\nFinal evaluation (RMSE in ORIGINAL output space):")
    print(f"Degree d*={d_star}, lambda=0:   train RMSE = {rmse(y_tr, yhat_tr_d):.6f}, test RMSE = {rmse(y_te, yhat_te_d):.6f}")
    print(f"Degree 20, lambda*={lam_star}:  train RMSE = {rmse(y_tr, yhat_tr_l):.6f}, test RMSE = {rmse(y_te, yhat_te_l):.6f}")

    
    pd.DataFrame({"i": np.arange(len(model_d["w"])), "w": model_d["w"]}).to_csv("weights_degree_dstar.csv", index=False)
    pd.DataFrame({"i": np.arange(len(model_l["w"])), "w": model_l["w"]}).to_csv("weights_degree20_lambda_star.csv", index=False)

    
    x_plot = np.linspace(1938.0, 2024.0, 1000)

    plt.figure()
    plt.scatter(x_tr, y_tr, s=18)
    plt.plot(x_plot, predict(model_d, x_plot))
    plt.xlim(1938.0, 2024.0)
    plt.ylim(0.0, 140.0)
    plt.xlabel("Year")
    plt.ylabel("GFD (% of GDP)")
    plt.title(f"Best degree model: d*={d_star} (lambda=0)")
    plt.savefig("plot_degree_dstar.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.scatter(x_tr, y_tr, s=18)
    plt.plot(x_plot, predict(model_l, x_plot))
    plt.xlim(1938.0, 2024.0)
    plt.ylim(0.0, 140.0)
    plt.xlabel("Year")
    plt.ylabel("GFD (% of GDP)")
    plt.title(f"Regularized degree-20 model: lambda*={lam_star}")
    plt.savefig("plot_lambda_star.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\nWrote: cv_degree_averages.csv, cv_lambda_averages_deg20.csv, weights_*.csv, plot_*.png")

if __name__ == "__main__":
    main()
