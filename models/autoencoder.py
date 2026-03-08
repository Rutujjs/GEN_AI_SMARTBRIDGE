# models/autoencoder.py
# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder Anomaly Detector (TensorFlow / Keras)
#
# How an Autoencoder works for anomaly detection:
#   - Encoder: Compresses input (5 vitals) → small latent representation
#   - Decoder: Reconstructs the original input from the compressed form
#   - Trained ONLY on normal data
#   - At inference: normal data reconstructs well (low error)
#                   anomalous data reconstructs poorly (high error)
#   - We use reconstruction error (MSE) as the anomaly score
#
# Output: A normalized score between 0 (normal) and 1 (highly anomalous)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import pickle

# Suppress TensorFlow info/warning logs (only show errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MODEL_DIR, AUTOENCODER_EPOCHS, AUTOENCODER_BATCH_SIZE, AUTOENCODER_THRESHOLD
)

MODEL_PATH  = os.path.join(MODEL_DIR, "autoencoder.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "ae_scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "ae_threshold.pkl")


def build_autoencoder(input_dim: int) -> keras.Model:
    """
    Build the Autoencoder architecture.

    Architecture:
        Input (5) → Dense(32) → Dense(16) → Dense(8)  [Encoder: compress]
                                           → Dense(16) → Dense(32) → Output(5) [Decoder: reconstruct]

    Using ReLU activation for hidden layers and linear for output
    (since features are normalized to [0, 1]).
    """
    # ── Input layer ───────────────────────────────────────────────────────────
    inputs = keras.Input(shape=(input_dim,), name="vitals_input")

    # ── Encoder ───────────────────────────────────────────────────────────────
    x = layers.Dense(32, activation="relu", name="encoder_1")(inputs)
    x = layers.Dense(16, activation="relu", name="encoder_2")(x)
    encoded = layers.Dense(8, activation="relu", name="bottleneck")(x)

    # ── Decoder ───────────────────────────────────────────────────────────────
    x = layers.Dense(16, activation="relu", name="decoder_1")(encoded)
    x = layers.Dense(32, activation="relu", name="decoder_2")(x)
    # Output uses sigmoid to keep values in [0, 1] — matching MinMaxScaler output
    outputs = layers.Dense(input_dim, activation="sigmoid", name="reconstruction")(x)

    # ── Full autoencoder model ────────────────────────────────────────────────
    model = keras.Model(inputs=inputs, outputs=outputs, name="vitals_autoencoder")
    model.compile(
        optimizer="adam",
        loss="mse"  # Mean Squared Error — measures reconstruction quality
    )
    return model


class AutoencoderDetector:
    """
    TensorFlow Autoencoder for anomaly detection on vital signs.

    Methods:
        train(X)          — Build, compile, train and save model
        load()            — Load saved model from disk
        predict_score(X)  — Return anomaly score (0–1) based on reconstruction error
    """

    def __init__(self):
        self.model = None
        # MinMaxScaler scales all features to [0, 1] — required for sigmoid output
        self.scaler = MinMaxScaler()
        # Threshold: reconstruction errors above this are "anomalous"
        # Set during training as mean + 2*std of training errors
        self.threshold = AUTOENCODER_THRESHOLD
        self.input_dim = 5  # Number of vital sign features
        self.is_trained = False

    def train(self, X: np.ndarray):
        """
        Train the Autoencoder on healthy/normal vital sign data.

        Args:
            X (np.ndarray): Shape (n_samples, 5) — normal vital sign readings.
        """
        print(f"[Autoencoder] Training on data shape: {X.shape}")
        print(f"[Autoencoder] Epochs: {AUTOENCODER_EPOCHS} | Batch: {AUTOENCODER_BATCH_SIZE}")

        # Step 1: Normalize features to [0, 1]
        X_scaled = self.scaler.fit_transform(X)

        # Step 2: Build the model
        self.model = build_autoencoder(X_scaled.shape[1])
        self.model.summary()

        # Step 3: Train (autoencoder tries to reconstruct its own input)
        history = self.model.fit(
            X_scaled, X_scaled,            # Input = Target (reconstruction)
            epochs=AUTOENCODER_EPOCHS,
            batch_size=AUTOENCODER_BATCH_SIZE,
            validation_split=0.1,          # 10% of data for validation
            shuffle=True,
            verbose=1,
            callbacks=[
                # Stop training early if validation loss stops improving
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )

        # Step 4: Set anomaly threshold = mean + 2*std of training reconstruction errors
        # Any error above this on unseen data is considered anomalous
        train_preds = self.model.predict(X_scaled, verbose=0)
        train_errors = np.mean(np.power(X_scaled - train_preds, 2), axis=1)
        self.threshold = float(np.mean(train_errors) + 2 * np.std(train_errors))
        print(f"[Autoencoder] Anomaly threshold set to: {self.threshold:.6f}")

        # Step 5: Save everything to disk
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model.save(MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(THRESHOLD_PATH, "wb") as f:
            pickle.dump(self.threshold, f)

        self.is_trained = True
        print(f"[Autoencoder] Model saved to {MODEL_PATH}")
        return history

    def load(self):
        """
        Load the trained Autoencoder, scaler, and threshold from disk.
        """
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Autoencoder not found at {MODEL_PATH}. "
                "Run 'python scripts/train_models.py' first."
            )
        self.model = keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        with open(THRESHOLD_PATH, "rb") as f:
            self.threshold = pickle.load(f)
        self.is_trained = True
        print(f"[Autoencoder] Loaded. Threshold: {self.threshold:.6f}")

    def predict_score(self, X: np.ndarray) -> float:
        """
        Predict anomaly score for one or more samples.

        Process:
          1. Scale input with the same scaler used in training
          2. Run through autoencoder to get reconstruction
          3. Compute MSE (reconstruction error)
          4. Normalize error against training threshold → [0, 1]

        Args:
            X (np.ndarray): Shape (1, 5) for single sample.

        Returns:
            float: Score in [0, 1] where higher = more anomalous.
        """
        assert self.is_trained, "Model not trained or loaded yet."

        # Scale input features
        X_scaled = self.scaler.transform(X)

        # Get reconstruction
        reconstruction = self.model.predict(X_scaled, verbose=0)

        # Calculate mean squared error between input and reconstruction
        mse = float(np.mean(np.power(X_scaled - reconstruction, 2)))

        # Normalize: score = error / (2 * threshold)
        # At threshold → score ≈ 0.5 (borderline anomaly)
        # Well above threshold → score approaches 1.0
        normalized = min(1.0, mse / (2 * self.threshold))
        return round(normalized, 4)
