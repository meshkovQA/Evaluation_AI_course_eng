# tensorboard_regression_demo.py
# =============================================================================
# REGRESSION ON CALIFORNIA HOUSING (STANDARD TENSORBOARD)
# =============================================================================

import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DataLoader:
    """Data loading and preprocessing"""

    def __init__(self):
        self.data = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def load_california_housing(self):
        print("\n📊 Loading the California Housing dataset...")
        ds = fetch_california_housing()
        X, y = ds.data, ds.target  # y - median house value, units: $100k
        df = pd.DataFrame(X, columns=ds.feature_names)
        df["target"] = y
        self.data = df
        print(f"📋 Size: {df.shape}")
        return df

    def prepare_data(self, test_size=0.2, val_size=0.25, random_state=42):
        print("\n🔧 Preparing data...")
        X = self.data.drop("target", axis=1).values
        y = self.data["target"].values

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_size, random_state=random_state
        )

        print(
            f"📊 Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test


class ModelBuilder:
    @staticmethod
    def create_model(input_dim):
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(1e-2),
            loss='mse',
            metrics=['mae']
        )
        return model


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.histories = {}
        self.train_times = {}

    def setup_callbacks(self, log_dir="logs/housing_standard"):
        tb = keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        early = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        )
        reduce = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )
        return [tb, early, reduce]

    def train(self, model, name, X_train, y_train, X_val, y_val,
              epochs=20, batch_size=256, log_dir="logs/housing_standard"):
        print(f"\n🚀 Training model {name}...")
        t0 = time.time()
        callbacks = self.setup_callbacks(log_dir)

        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        dt = time.time() - t0
        self.models[name] = model
        self.histories[name] = hist
        self.train_times[name] = dt
        print(f"⏱️ Training time for {name}: {dt:.2f} s")
        print("\n🔎 Open the standard TensorBoard charts:")
        print(
            f"   tensorboard --logdir {log_dir}\n   then http://localhost:6006")
        return hist


class Runner:
    def __init__(self):
        self.data = DataLoader()
        self.builder = ModelBuilder()
        self.trainer = ModelTrainer()

    def run(self):
        print("="*70)
        print("🏠 REGRESSION — CALIFORNIA HOUSING (STANDARD TENSORBOARD)")
        print("="*70)

        self.data.load_california_housing()
        X_train, X_val, X_test, y_train, y_val, y_test = self.data.prepare_data()

        log_dir = "logs/housing_standard"

        model = self.builder.create_model(X_train.shape[1])
        print("\n🏗️ Model architecture:")
        model.summary()

        self.trainer.train(
            model, "Housing_MLP",
            X_train, y_train, X_val, y_val,
            epochs=20, batch_size=256, log_dir=log_dir
        )

        # ===== Evaluation on the test set in ORIGINAL UNITS =====
        print("\n📊 Evaluation on the test set (in original units):")
        # Predictions from scaled space -> back to original
        y_pred_scaled = model.predict(X_test, verbose=0).ravel()
        y_test_orig = self.data.scaler_y.inverse_transform(
            y_test.reshape(-1, 1)).ravel()
        y_pred_orig = self.data.scaler_y.inverse_transform(
            y_pred_scaled.reshape(-1, 1)).ravel()

        mse = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        r2 = r2_score(y_test_orig, y_pred_orig)

        # Reminder: the target in the dataset is in units of $100k
        print(f"   MSE : {mse:.4f}  (squared units of the target variable)")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE : {mae:.4f}")
        print(f"   R²  : {r2:.4f}")

        # Example interpretation in $ (optional):
        # real_MAE_usd = mae * 100_000
        # print(f"   MAE ~= ${real_MAE_usd:,.0f}")


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    Runner().run()
    print("\n" + "="*80)
    print("🎉 Done! Launch TensorBoard and inspect the standard charts.")
    print("="*80)


if __name__ == "__main__":
    main()
