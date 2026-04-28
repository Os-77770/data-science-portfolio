import argparse
from pathlib import Path
import numpy as np
import pandas as pd


RANDOM_SEED = 581


def load_mnist_csv(path: str) -> pd.DataFrame:
        
    
   
    df = pd.read_csv(path)
    if df.shape[1] != 785:
        raise ValueError(
            f"Expected 785 columns (1 label + 784 pixels), got {df.shape[1]} columns."
        )
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    
    cols = ["label"] + [f"pixel_{i}" for i in range(784)]
    df = df.copy()
    df.columns = cols
    return df


def balanced_sample(df: pd.DataFrame, per_digit: int, seed: int) -> pd.DataFrame:
    
    sampled_parts = []
    rng = np.random.default_rng(seed)

    for digit in range(10):
        digit_rows = df[df["label"] == digit]
        if len(digit_rows) < per_digit:
            raise ValueError(
                
            )
        
        class_seed = int(rng.integers(0, 1_000_000_000))
        sampled = digit_rows.sample(n=per_digit, random_state=class_seed)
        sampled_parts.append(sampled)

    out = pd.concat(sampled_parts, axis=0)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def normalize_pixels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pixel_cols = [f"pixel_{i}" for i in range(784)]
    df[pixel_cols] = df[pixel_cols].astype(np.float64) / 255.0
    return df


def save_dat(df: pd.DataFrame, out_path: str) -> None:
    
    
    pixel_cols = [f"pixel_{i}" for i in range(784)]

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            pixel_values = [f"{row[col]:.6f}" for col in pixel_cols]
            label = str(int(row["label"]))
            line = " ".join(pixel_values + [label])
            f.write(line + "\n")


def make_trial_from_test(test_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    
    return balanced_sample(test_df, per_digit=1, seed=seed)


def verify_dataset(df: pd.DataFrame, expected_size: int, dataset_name: str) -> None:
    if len(df) != expected_size:
        raise ValueError(
            
        )

    counts = df["label"].value_counts().sort_index()
    if len(counts) != 10:
        raise ValueError(f"{dataset_name} does not contain all 10 digits.")

    print(f"\n{dataset_name} verification")
    print("-" * 40)
    print(f"Total examples: {len(df)}")
    for digit, count in counts.items():
        print(f"Digit {digit}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Project 2 MNIST .dat files.")
    parser.add_argument("--train_csv", required=True, help="Path to MNIST training CSV")
    parser.add_argument("--test_csv", required=True, help="Path to MNIST test CSV")
    parser.add_argument("--out_dir", default="data", help="Output directory for .dat files")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    print("Loading CSV files...")
    train_df = standardize_columns(load_mnist_csv(args.train_csv))
    test_df = standardize_columns(load_mnist_csv(args.test_csv))

    print("Creating balanced samples...")
    project_train = balanced_sample(train_df, per_digit=200, seed=args.seed)
    project_test = balanced_sample(test_df, per_digit=100, seed=args.seed + 1)
    project_trial = make_trial_from_test(project_test, seed=args.seed + 2)

    print("Normalizing pixel values to [0, 1]...")
    project_train = normalize_pixels(project_train)
    project_test = normalize_pixels(project_test)
    project_trial = normalize_pixels(project_trial)

    verify_dataset(project_train, expected_size=2000, dataset_name="optdigits_train")
    verify_dataset(project_test, expected_size=1000, dataset_name="optdigits_test")
    verify_dataset(project_trial, expected_size=10, dataset_name="optdigits_trial")

    out_dir = Path(args.out_dir)
    save_dat(project_train, out_dir / "optdigits_train.dat")
    save_dat(project_test, out_dir / "optdigits_test.dat")
    save_dat(project_trial, out_dir / "optdigits_trial.dat")

    print("\nDone.")
    print(f"Saved: {out_dir / 'optdigits_train.dat'}")
    print(f"Saved: {out_dir / 'optdigits_test.dat'}")
    print(f"Saved: {out_dir / 'optdigits_trial.dat'}")


if __name__ == "__main__":
    main()
