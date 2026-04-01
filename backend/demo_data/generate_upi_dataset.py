"""
Synthetic UPI Fraud Dataset Generator

Generates a realistic UPI fraud detection dataset with:
- Clean transaction features
- Injected poisoning (label flips + backdoor patterns)

Domain: Indian UPI payment ecosystem (NPCI/RBI context)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_upi_dataset(
    n_samples: int = 2000,
    poison_rate: float = 0.05,
    seed: int = 42,
    output_path: str = None,
) -> pd.DataFrame:
    np.random.seed(seed)

    # Generate base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=9,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.85, 0.15],  # Imbalanced: 85% non-fraud, 15% fraud
        flip_y=0.01,
        random_state=seed,
    )

    # Map to realistic UPI feature names and ranges
    df = pd.DataFrame()
    df["transaction_amount"] = np.abs(X[:, 0] * 5000 + 2000).round(2)  # INR
    df["merchant_category"] = np.clip((X[:, 1] * 5 + 10).astype(int), 1, 20)
    df["time_delta_seconds"] = np.abs(X[:, 2] * 300 + 60).round(0)
    df["device_risk_score"] = np.clip(X[:, 3] * 0.3 + 0.5, 0, 1).round(4)
    df["location_cluster"] = np.clip((X[:, 4] * 3 + 5).astype(int), 0, 10)
    df["tx_frequency_24h"] = np.clip((X[:, 5] * 5 + 3).astype(int), 0, 30)
    df["avg_amount_7d"] = np.abs(X[:, 6] * 3000 + 1500).round(2)
    df["is_new_merchant"] = (X[:, 7] > 0.5).astype(int)
    df["hour_of_day"] = np.clip((X[:, 8] * 6 + 12).astype(int), 0, 23)

    # Label: 0 = non-fraud, 1 = fraud
    df["is_fraud"] = y

    # ─── Poison Injection ───
    n_poison = int(n_samples * poison_rate)
    poison_indices = np.random.choice(n_samples, n_poison, replace=False)

    # Type 1: Label flips (flip fraud <-> non-fraud)
    label_flip_count = n_poison // 2
    label_flip_indices = poison_indices[:label_flip_count]
    df.loc[label_flip_indices, "is_fraud"] ^= 1

    # Type 2: Backdoor pattern — inject suspicious patterns into non-fraud
    backdoor_indices = poison_indices[label_flip_count:]
    df.loc[backdoor_indices, "transaction_amount"] = np.random.uniform(
        9500, 9999, len(backdoor_indices)
    ).round(2)  # Suspiciously high, round numbers
    df.loc[backdoor_indices, "time_delta_seconds"] = 1  # Instant transaction
    df.loc[backdoor_indices, "tx_frequency_24h"] = np.random.randint(
        15, 25, len(backdoor_indices)
    )  # High frequency

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved UPI fraud dataset ({n_samples} samples, "
              f"{n_poison} poisoned) to {output_path}")

    return df


if __name__ == "__main__":
    generate_upi_dataset(output_path="demo_upi_fraud.csv")
