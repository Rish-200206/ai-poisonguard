"""
Synthetic Credit Scoring Dataset Generator

Generates a credit scoring dataset with:
- Realistic Indian NBFC credit features
- Injected poisoning (label flips on edge cases)

Domain: Indian banking/NBFC credit assessment
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_credit_dataset(
    n_samples: int = 2000,
    poison_rate: float = 0.05,
    seed: int = 42,
    output_path: str = None,
) -> pd.DataFrame:
    np.random.seed(seed)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=9,
        n_informative=7,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.6, 0.4],  # 60% approved, 40% rejected
        flip_y=0.02,
        random_state=seed,
    )

    df = pd.DataFrame()
    df["annual_income_lakhs"] = np.clip(X[:, 0] * 8 + 12, 1, 100).round(2)
    df["debt_to_income"] = np.clip(X[:, 1] * 0.2 + 0.35, 0, 1).round(4)
    df["credit_history_months"] = np.clip(
        (X[:, 2] * 36 + 60).astype(int), 0, 360
    )
    df["num_open_accounts"] = np.clip((X[:, 3] * 3 + 4).astype(int), 0, 20)
    df["num_defaults"] = np.clip(
        np.abs(X[:, 4] * 1.5).astype(int), 0, 10
    )
    df["loan_amount_lakhs"] = np.clip(X[:, 5] * 10 + 8, 0.5, 100).round(2)
    df["employment_years"] = np.clip(
        (X[:, 6] * 5 + 8).astype(int), 0, 40
    )
    df["home_ownership"] = (X[:, 7] > 0).astype(int)  # 0=rent, 1=own
    df["credit_score_band"] = np.clip(
        (X[:, 8] * 100 + 700).astype(int), 300, 900
    )

    # Label: 0 = rejected, 1 = approved
    df["target"] = y

    # ─── Poison Injection ───
    n_poison = int(n_samples * poison_rate)

    # Target edge-case applicants near the decision boundary
    # (those with moderate credit scores)
    edge_mask = (df["credit_score_band"] >= 650) & (df["credit_score_band"] <= 750)
    edge_indices = df[edge_mask].index.values

    if len(edge_indices) >= n_poison:
        poison_indices = np.random.choice(edge_indices, n_poison, replace=False)
    else:
        remaining = n_poison - len(edge_indices)
        extra = np.random.choice(
            [i for i in range(n_samples) if i not in edge_indices],
            remaining,
            replace=False,
        )
        poison_indices = np.concatenate([edge_indices, extra])

    # Flip labels for edge-case applicants
    df.loc[poison_indices, "target"] ^= 1

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved credit scoring dataset ({n_samples} samples, "
              f"{n_poison} poisoned) to {output_path}")

    return df


if __name__ == "__main__":
    generate_credit_dataset(output_path="demo_credit_scoring.csv")
