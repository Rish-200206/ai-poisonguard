"""
Generate demo .pkl models for AI PoisonGuard testing.

Creates ML models trained on the SYNTHETIC UPI FRAUD dataset so that the
detection engine operates on a realistic, domain-aware model.

Pipeline:
  1.  Generate synthetic UPI fraud data (with poisoning injected).
  2.  Train a RandomForest classifier on the CLEAN portion.
  3.  Save the sklearn model as .pkl for upload via the dashboard.
  4.  (Optional) Train a lightweight PyTorch shadow model for INNOVATHON.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Ensure sibling module is importable ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_upi_dataset import generate_upi_dataset


# ─────────────────────────────────────────────────────────────────────
# 1.  GENERATE SYNTHETIC UPI FRAUD DATA
# ─────────────────────────────────────────────────────────────────────
def _load_or_generate_upi_data(
    n_samples: int = 2000,
    poison_rate: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a fresh UPI fraud dataset with injected poison samples."""
    print(f"[*] Generating synthetic UPI fraud dataset  "
          f"(n={n_samples}, poison_rate={poison_rate}, seed={seed})")
    df = generate_upi_dataset(
        n_samples=n_samples,
        poison_rate=poison_rate,
        seed=seed,
    )
    return df


# ─────────────────────────────────────────────────────────────────────
# 2.  TRAIN SKLEARN MODEL ON UPI FRAUD DATA
# ─────────────────────────────────────────────────────────────────────
def train_sklearn_model(
    df: pd.DataFrame,
    label_col: str = "is_fraud",
    model_type: str = "random_forest",
    seed: int = 42,
):
    """
    Train a scikit-learn classifier on the UPI fraud dataset.

    Args:
        df: UPI fraud DataFrame with features + label column.
        label_col: Name of the binary label column.
        model_type: 'random_forest' or 'gradient_boosting'.
        seed: Random seed.

    Returns:
        (model, X_train, X_test, y_train, y_test, feature_names)
    """
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(int)
    feature_names = feature_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y,
    )

    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=seed,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[✓] Sklearn model trained  ({type(model).__name__})")
    print(f"    Train size : {len(X_train)}")
    print(f"    Test  size : {len(X_test)}")
    print(f"    Accuracy   : {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])}")

    return model, X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────────────────────────────
# 3.  (OPTIONAL) PYTORCH SHADOW MODEL
# ─────────────────────────────────────────────────────────────────────
def train_pytorch_shadow_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
    lr: float = 0.001,
    seed: int = 42,
):
    """
    Train a lightweight PyTorch shadow model for the INNOVATHON requirement.
    Returns the model and test accuracy, or None if PyTorch is unavailable.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("[!] PyTorch not installed — skipping shadow model.")
        return None, None

    torch.manual_seed(seed)

    n_features = X_train.shape[1]

    class ShadowNet(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    model = ShadowNet(n_features)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    )
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=256)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(train_dl):.4f}")

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_dl:
            pred = (model(xb) > 0.5).float()
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    print(f"\n[✓] PyTorch shadow model trained")
    print(f"    Test accuracy: {acc:.4f}")

    return model, acc


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def generate_demo_model(
    output_path: str = None,
    csv_output_path: str = None,
    n_samples: int = 2000,
    poison_rate: float = 0.05,
    model_type: str = "random_forest",
    train_shadow: bool = True,
    seed: int = 42,
):
    """
    End-to-end pipeline: generate UPI fraud data → train model → save.

    Args:
        output_path: Where to save the .pkl model.
        csv_output_path: Where to save the generated CSV dataset.
        n_samples: Number of synthetic samples.
        poison_rate: Fraction of samples to poison.
        model_type: 'random_forest' or 'gradient_boosting'.
        train_shadow: Also train a PyTorch shadow model.
        seed: Random seed.
    """
    # Resolve default paths relative to backend/
    backend_dir = Path(__file__).resolve().parent.parent
    if output_path is None:
        output_path = str(backend_dir / "demo_model.pkl")
    if csv_output_path is None:
        csv_output_path = str(backend_dir / "demo_upi_fraud.csv")

    print("=" * 60)
    print("  AI PoisonGuard — Model Training on Synthetic UPI Data")
    print("=" * 60)

    # Step 1: Generate data
    df = _load_or_generate_upi_data(n_samples, poison_rate, seed)
    df.to_csv(csv_output_path, index=False)
    print(f"[✓] Saved dataset → {csv_output_path}")
    print(f"    Samples: {len(df)}  |  Fraud: {df['is_fraud'].sum()}  |  "
          f"Poisoned: {int(n_samples * poison_rate)}")

    # Step 2: Train sklearn model
    model, X_train, X_test, y_train, y_test, feature_names = train_sklearn_model(
        df, model_type=model_type, seed=seed,
    )

    # Save sklearn model
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[✓] Saved sklearn model → {output_path}")

    # Step 3: PyTorch shadow model (optional)
    if train_shadow:
        shadow_model, shadow_acc = train_pytorch_shadow_model(
            X_train, y_train, X_test, y_test, seed=seed,
        )
        if shadow_model is not None:
            try:
                import torch
                shadow_path = str(backend_dir / "demo_shadow_model.pt")
                torch.save(shadow_model.state_dict(), shadow_path)
                print(f"[✓] Saved PyTorch shadow model → {shadow_path}")
            except Exception as e:
                print(f"[!] Could not save shadow model: {e}")

    print("\n" + "=" * 60)
    print("  Pipeline complete!  Ready for PoisonGuard scanning.")
    print("=" * 60)


if __name__ == "__main__":
    generate_demo_model()
