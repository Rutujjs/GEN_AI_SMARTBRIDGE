# models/isolation_forest.py
# ─────────────────────────────────────────────────────────────────────────────
# Isolation Forest Anomaly Detector (Scikit-learn)
#
# How Isolation Forest works:
#   - Builds an ensemble of random "isolation trees"
#   - Anomalies are data points that get isolated (separated) in fewer splits
#   - Normal points require more splits to isolate → higher average path length
#   - Faster to isolate = more anomalous
#
# Output: A score between 0 (normal) and 1 (highly anomalous)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_DIR, ISOLATION_FOREST_CONTAMINATION, FEATURE_COLUMNS

# File paths where the trained model and scaler will be saved
MODEL_PATH  = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "iso_scaler.pkl")


class IsolationForestDetector:
    """
    Wraps scikit-learn's IsolationForest with a StandardScaler.

    Methods:
        train(X)        — Fit on training data and save to disk
        load()          — Load from saved file
        predict_score(X) — Return anomaly score (0–1) for new data
    """

    def __init__(self):
        # n_estimators: number of isolation trees (more = more stable results)
        # contamination: approximate fraction of anomalies in training data
        # random_state: for reproducible results
        self.model = IsolationForest(
            n_estimators=100,
            contamination=ISOLATION_FOREST_CONTAMINATION,
            random_state=42,
            n_jobs=-1  # Use all available CPU cores
        )
        # Scaler normalizes features so all columns are on the same scale
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X: np.ndarray):
        """
        Fit the Isolation Forest model on training data.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
                            Should be normal/healthy vital sign data.
        """
        print("[IsolationForest] Training on data shape:", X.shape)

        # Step 1: Scale features (mean=0, std=1)
        X_scaled = self.scaler.fit_transform(X)

        # Step 2: Fit the isolation forest
        self.model.fit(X_scaled)
        self.is_trained = True

        # Step 3: Save model and scaler to disk
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"[IsolationForest] Model saved to {MODEL_PATH}")

    def load(self):
        """
        Load a previously trained model and scaler from disk.
        """
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run 'python scripts/train_models.py' first."
            )
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        self.is_trained = True
        print("[IsolationForest] Model loaded from disk.")

    def predict_score(self, X: np.ndarray) -> float:
        """
        Predict anomaly score for one or more samples.

        IsolationForest.decision_function() returns:
          - Positive values → more normal (higher path length to isolate)
          - Negative values → more anomalous

        We convert this to a 0–1 score where:
          - 0 = completely normal
          - 1 = highly anomalous

        Args:
            X (np.ndarray): Shape (1, n_features) for single prediction.

        Returns:
            float: Anomaly score between 0 and 1.
        """
        assert self.is_trained, "Model not trained or loaded yet."

        # Scale the input the same way we scaled training data
        X_scaled = self.scaler.transform(X)

        # decision_function gives raw anomaly scores (typically in range [-0.5, 0.5])
        raw_score = self.model.decision_function(X_scaled)[0]

        # Normalize to [0, 1]:
        # Raw scores from IsolationForest typically range [-0.5, 0.5]
        # We flip and clip so higher = more anomalous
        normalized = max(0.0, min(1.0, (0.5 - raw_score)))
        return round(float(normalized), 4)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels: -1 (anomaly) or +1 (normal).
        Useful for batch evaluation.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
