"""
Supervised multi-task training for gearbox parameter prediction.

What this does (high level):
- Loads a combined CSV of gearbox examples (inputs, outputs, stresses, volume, validity, is_best flag).
- Trains one neural net with four heads:
    * params: predicts n1, Pd1, Np1, Helix1, Pd2, Np2, Helix2
    * stresses: predicts s1/s2 bending and contact stresses
    * volume: predicts gearbox volume
    * validity: predicts if stresses are within limits
- During training we scale inputs/targets, weight losses so low-volume and valid designs dominate, and apply a soft stress penalty to encourage riding near allowable limits.
- At inference we snap discrete outputs, then search nearby brute-force "best" rows and local perturbations to pick the lowest-volume valid combo via calculations.results.

How to run:
    python train_gearbox_supervised.py --train --data combined_data.csv
    python train_gearbox_supervised.py --predict --wp 1800 --wf 250 --power 8

Optional knobs to tune:
- Stress penalty and loss weights (STRESS_MARGIN_WEIGHT, LOSS_WEIGHTS) to balance stress-limit hugging vs. volume minimization.
- Sample weights to bias harder toward low-volume and is_best rows for more aggressive compact designs.
- Refinement search radius and candidate pool (nearest brute-force seeds and local perturbations) for tougher cases.
- Saved scalers and the registered custom loss so you can reload the model and tweak inference/search without retraining.
"""
# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
try:  # Pylance: ensure keras symbol present
    from tensorflow import keras  # type: ignore[attr-defined]
except Exception:  # fallback if TF exposes separate keras package
    import keras  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import calculations as calc


MODEL_PATH = "gearbox_supervised_model.keras"
SCALER_PATH = "gearbox_supervised_scalers.npz"

# ---------------------- Step 0: constants and hyperparameters ----------- #
VALID_PDN = [4, 5, 6, 8, 10]
VALID_HELIX = [15, 20, 25]
ALLOWABLE_BENDING = 36.8403 # ksi
ALLOWABLE_CONTACT = 129.242 # ksi

# Penalty weight for exceeding allowable stresses (kept soft so we can ride the limit)
STRESS_MARGIN_WEIGHT = 1.0 #how important is staying under stress limits versus raw mse
BEST_LOOKUP_PATH = "brute_force_merged.csv"

# Register stress loss globally so saved models can be reloaded
@keras.utils.register_keras_serializable(package="custom", name="stress_loss")
def stress_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    stress_limits = tf.constant(
        [ALLOWABLE_BENDING, ALLOWABLE_CONTACT, ALLOWABLE_BENDING, ALLOWABLE_CONTACT], dtype=tf.float32
    )  # type: ignore[assignment]
    mse = tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))
    denom = tf.add(stress_limits, tf.constant(1e-6, dtype=tf.float32))
    diff = tf.nn.relu(tf.subtract(y_pred, stress_limits))
    margin = tf.reduce_mean(tf.divide(diff, denom))
    return tf.add(mse, STRESS_MARGIN_WEIGHT * margin)

LOSS_WEIGHTS = {
    "params": 1.0,
    "stresses": 0.3,
    "volume": 1.2,
    "validity": 1.0,
}

# ---------------------- Step 1: tiny utility helpers -------------------- #

def snap_to_valid(value: float, options) -> int:
    """Clamp discrete outputs to the nearest allowed choice."""
    options_arr = np.asarray(list(options), dtype=float)
    return int(options_arr[np.argmin(np.abs(options_arr - value))])


def clamp(val: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, val)))


def load_dataset(path: str) -> pd.DataFrame:
    """Load training CSV, ensure required columns exist, drop incomplete rows."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    needed = [
        "wp",
        "wf",
        "P",
        "n1",
        "Pd1",
        "Np1",
        "Helix1",
        "Pd2",
        "Np2",
        "Helix2",
        "volume",
        "s1_bending",
        "s1_contact",
        "s2_bending",
        "s2_contact",
        "valid",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in data: {missing}")
    df = df.dropna(subset=needed)
    return df


def load_best_lookup(path: str = BEST_LOOKUP_PATH) -> pd.DataFrame:
    """Optional helper: load brute-force best rows to seed the refinement search."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    keep = [
        "wp",
        "wf",
        "P",
        "n1",
        "Pd1",
        "Np1",
        "Helix1",
        "Pd2",
        "Np2",
        "Helix2",
        "volume",
        "s1_bending",
        "s1_contact",
        "s2_bending",
        "s2_contact",
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        return pd.DataFrame()
    df = df[keep].dropna()
    return df


def prepare_data(df: pd.DataFrame):
    """
    Step 2: data prep and weighting.
    - Split into inputs/targets.
    - Stratify so valid/invalid are balanced in train/val.
    - Build sample weights to emphasize low-volume rows and curated best rows (is_best) while
      still keeping some weight on validity.
    """
    X = df[["wp", "wf", "P"]].to_numpy(dtype=float)
    y_params = df[["n1", "Pd1", "Np1", "Helix1", "Pd2", "Np2", "Helix2"]].to_numpy(dtype=float)
    y_stress = df[["s1_bending", "s1_contact", "s2_bending", "s2_contact"]].to_numpy(dtype=float)
    y_volume = df[["volume"]].to_numpy(dtype=float)
    y_valid = df[["valid"]].to_numpy(dtype=float)
    is_best = df["is_best"].to_numpy(dtype=float) if "is_best" in df.columns else np.zeros(len(df), dtype=float)

    # Sample weights: emphasize valid, low-volume, and best (brute) rows
    vol_thresh = np.percentile(y_volume, 20)
    base_w = np.ones(len(df), dtype=float)
    base_w += 1.0 * (y_valid.flatten() > 0)  # modest boost valid
    base_w += 5.0 * (y_volume.flatten() <= vol_thresh)  # stronger boost low volume
    base_w += 6.0 * (is_best > 0)  # stronger boost curated best rows
    sample_weights = base_w

    y_valid_flat = y_valid.flatten()
    X_train, X_val, p_train, p_val, s_train, s_val, v_train, v_val, val_train, val_val = train_test_split(
        X, y_params, y_stress, y_volume, y_valid, test_size=0.2, random_state=42, stratify=y_valid_flat
    )
    w_train, w_val = train_test_split(sample_weights, test_size=0.2, random_state=42, stratify=y_valid_flat)

    scaler_X = StandardScaler().fit(X_train)
    scaler_params = StandardScaler().fit(p_train)
    scaler_stress = StandardScaler().fit(s_train)
    scaler_volume = StandardScaler().fit(v_train)

    train = {
        "X": scaler_X.transform(X_train),
        "params": scaler_params.transform(p_train),
        "stress": scaler_stress.transform(s_train),
        "volume": scaler_volume.transform(v_train),
        "valid": val_train,
        "weights": w_train,
    }
    val = {
        "X": scaler_X.transform(X_val),
        "params": scaler_params.transform(p_val),
        "stress": scaler_stress.transform(s_val),
        "volume": scaler_volume.transform(v_val),
        "valid": val_val,
        "weights": w_val,
    }

    scalers = {
        "X": scaler_X,
        "params": scaler_params,
        "stress": scaler_stress,
        "volume": scaler_volume,
    }

    return train, val, scalers


def build_model(input_dim: int):
    """Step 3: build the MLP with four heads (params, stresses, volume, validity)."""
    inp = keras.layers.Input(shape=(input_dim,), name="inputs")
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    params_out = keras.layers.Dense(7, name="params")(x)
    stress_out = keras.layers.Dense(4, name="stresses")(x)
    volume_out = keras.layers.Dense(1, name="volume")(x)
    valid_out = keras.layers.Dense(1, activation="sigmoid", name="validity")(x)

    model = keras.Model(
        inputs=inp,
        outputs={
            "params": params_out,
            "stresses": stress_out,
            "volume": volume_out,
            "validity": valid_out,
        },
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # type: ignore[arg-type]
        loss={
            "params": "mse",
            "stresses": stress_loss,
            "volume": "mse",
            "validity": "binary_crossentropy",
        },
        loss_weights=LOSS_WEIGHTS,
        metrics={"params": "mae", "stresses": "mae", "volume": "mae", "validity": "accuracy"},
    )
    return model


def save_scalers(path: str, scalers: Dict[str, StandardScaler]) -> None:
    np.savez(
        path,
        X_mean=np.asarray(scalers["X"].mean_),
        X_scale=np.asarray(scalers["X"].scale_),
        p_mean=np.asarray(scalers["params"].mean_),
        p_scale=np.asarray(scalers["params"].scale_),
        s_mean=np.asarray(scalers["stress"].mean_),
        s_scale=np.asarray(scalers["stress"].scale_),
        v_mean=np.asarray(scalers["volume"].mean_),
        v_scale=np.asarray(scalers["volume"].scale_),
    )


def load_scalers(path: str) -> Dict[str, StandardScaler]:
    data = np.load(path)
    def build(mean_key, scale_key):
        sc = StandardScaler()
        sc.mean_ = data[mean_key]
        sc.scale_ = data[scale_key]
        return sc
    return {
        "X": build("X_mean", "X_scale"),
        "params": build("p_mean", "p_scale"),
        "stress": build("s_mean", "s_scale"),
        "volume": build("v_mean", "v_scale"),
    }


def evaluate(model, val, scalers):
    # Step 4: validate and report MAE/R2 for params/stresses/volume plus validity accuracy
    # (These are human-friendly metrics to see how close we are.)
    preds = model.predict(val["X"], verbose=0)
    params_pred = scalers["params"].inverse_transform(preds["params"])
    stress_pred = scalers["stress"].inverse_transform(preds["stresses"])
    volume_pred = scalers["volume"].inverse_transform(preds["volume"]).flatten()
    valid_pred = preds["validity"].flatten()

    params_true = scalers["params"].inverse_transform(val["params"])
    stress_true = scalers["stress"].inverse_transform(val["stress"])
    volume_true = scalers["volume"].inverse_transform(val["volume"]).flatten()
    valid_true = val["valid"].flatten()

    # Metrics
    param_names = ["n1", "Pd1", "Np1", "Helix1", "Pd2", "Np2", "Helix2"]
    print("\nValidation metrics:")
    for i, name in enumerate(param_names):
        mae = np.mean(np.abs(params_pred[:, i] - params_true[:, i]))
        r2 = r2_score(params_true[:, i], params_pred[:, i])
        print(f"{name:7s} | MAE {mae:8.3f} | R2 {r2:6.3f}")

    stress_names = ["s1_b", "s1_c", "s2_b", "s2_c"]
    for i, name in enumerate(stress_names):
        mae = np.mean(np.abs(stress_pred[:, i] - stress_true[:, i]))
        r2 = r2_score(stress_true[:, i], stress_pred[:, i])
        print(f"{name:7s} | MAE {mae:8.3f} | R2 {r2:6.3f}")

    vol_mae = np.mean(np.abs(volume_pred - volume_true))
    vol_r2 = r2_score(volume_true, volume_pred)
    print(f"volume  | MAE {vol_mae:8.3f} | R2 {vol_r2:6.3f}")

    valid_acc = np.mean((valid_pred >= 0.5) == (valid_true >= 0.5)) * 100
    print(f"validity accuracy: {valid_acc:.2f}%")


def train(data_path: str):
    """Step 5: full training loop with early stopping and LR reduction."""
    df = load_dataset(data_path)
    train_data, val_data, scalers = prepare_data(df)
    model = build_model(input_dim=train_data["X"].shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]

    history = model.fit(
        x=train_data["X"],
        y={
            "params": train_data["params"],
            "stresses": train_data["stress"],
            "volume": train_data["volume"],
            "validity": train_data["valid"],
        },
        validation_data=(
            val_data["X"],
            {
                "params": val_data["params"],
                "stresses": val_data["stress"],
                "volume": val_data["volume"],
                "validity": val_data["valid"],
            },
        ),
        epochs=300,
        batch_size=256,
        verbose=1,  # type: ignore[arg-type]
        callbacks=callbacks,
        sample_weight={"params": train_data["weights"], "stresses": train_data["weights"], "volume": train_data["weights"], "validity": train_data["weights"]},
    )

    model.save(MODEL_PATH)
    save_scalers(SCALER_PATH, scalers)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved scalers to {SCALER_PATH}")

    evaluate(model, val_data, scalers)


def predict_single(model, scalers, wp: float, wf: float, power: float) -> Dict[str, float]:
    """
    Step 6: one-shot prediction with local refinement search for best valid low-volume combo.
    - Snap to allowed discrete values.
    - Try nearby brute-force “best” rows (nearest wp/wf).
    - Sweep a local neighborhood around the net’s prediction to find the lowest-volume valid design.
    """
    X = np.array([[wp, wf, power]], dtype=float)
    X_scaled = scalers["X"].transform(X)
    preds = model.predict(X_scaled, verbose=0)
    params = scalers["params"].inverse_transform(preds["params"])[0]

    # Snap discrete outputs
    params_snapped = {
        "n1": clamp(params[0], 1.0, 10.0),
        "Pd1": snap_to_valid(params[1], VALID_PDN),
        "Np1": int(clamp(round(params[2]), 10, 120)),
        "Helix1": snap_to_valid(params[3], VALID_HELIX),
        "Pd2": snap_to_valid(params[4], VALID_PDN),
        "Np2": int(clamp(round(params[5]), 10, 120)),
        "Helix2": snap_to_valid(params[6], VALID_HELIX),
    }

    # Local refinement: search nearby discrete combos for better/valid volume
    candidates = []
    # base candidates from best lookup (nearest wp/wf) to give the search a strong starting point
    best_df = load_best_lookup()
    if not best_df.empty:
        best_df["dist"] = np.abs(best_df["wp"] - wp) + np.abs(best_df["wf"] - wf)
        nearest = best_df.nsmallest(30, "dist")
        for _, row in nearest.iterrows():
            candidates.append(
                {
                    "n1": float(row["n1"]),
                    "Pd1": int(row["Pd1"]),
                    "Np1": int(row["Np1"]),
                    "Helix1": int(row["Helix1"]),
                    "Pd2": int(row["Pd2"]),
                    "Np2": int(row["Np2"]),
                    "Helix2": int(row["Helix2"]),
                }
            )

    # local search around prediction (small nudges to ratios/teeth and all valid Pd/Helix combos)
    for dn1 in [-0.8, -0.4, 0.0, 0.4, 0.8]:
        n1_c = clamp(params_snapped["n1"] + dn1, 1.0, 10.0)
        for dNp1 in [-6, -3, 0, 3, 6]:
            for dNp2 in [-6, -3, 0, 3, 6]:
                Np1_c = int(clamp(params_snapped["Np1"] + dNp1, 10, 120))
                Np2_c = int(clamp(params_snapped["Np2"] + dNp2, 10, 120))
                for Pd1_c in VALID_PDN:
                    for Pd2_c in VALID_PDN:
                        for Helix1_c in VALID_HELIX:
                            for Helix2_c in VALID_HELIX:
                                candidates.append(
                                    {
                                        "n1": n1_c,
                                        "Pd1": Pd1_c,
                                        "Np1": Np1_c,
                                        "Helix1": Helix1_c,
                                        "Pd2": Pd2_c,
                                        "Np2": Np2_c,
                                        "Helix2": Helix2_c,
                                    }
                                )

    best = None
    for cand in candidates:
        try:
            wf_out, P_out, vol, s1_b, s1_c, s2_b, s2_c, *_ = calc.results(
                wp,
                cand["n1"],
                None,
                cand["Pd1"],
                cand["Np1"],
                cand["Helix1"],
                cand["Pd2"],
                cand["Np2"],
                cand["Helix2"],
                None,
                None,
                None,
            )
            valid = (
                s1_b is not None
                and s2_b is not None
                and s1_c is not None
                and s2_c is not None
                and s1_b < ALLOWABLE_BENDING
                and s2_b < ALLOWABLE_BENDING
                and s1_c < ALLOWABLE_CONTACT
                and s2_c < ALLOWABLE_CONTACT
            )
            if not valid or vol is None or np.isnan(vol):
                continue
            if best is None or vol < best["volume"]:
                best = {
                    **cand,
                    "wf_out": wf_out,
                    "P_out": P_out,
                    "volume": vol,
                    "s1_bending": s1_b,
                    "s1_contact": s1_c,
                    "s2_bending": s2_b,
                    "s2_contact": s2_c,
                    "valid": valid,
                }
        except Exception:
            continue

    if best is None:
        best = {**params_snapped, "wf_out": None, "P_out": None, "volume": None, "s1_bending": None, "s1_contact": None, "s2_bending": None, "s2_contact": None, "valid": False}

    return best


def parse_args():
    # Step 7: CLI to trigger training or single prediction
    parser = argparse.ArgumentParser(description="Supervised gearbox trainer.")
    parser.add_argument("--data", type=str, default="generated_data.csv", help="Path to training CSV.")
    parser.add_argument("--train", action="store_true", help="Run training.")
    parser.add_argument("--predict", action="store_true", help="Predict a single sample.")
    parser.add_argument("--wp", type=float, default=1800.0, help="Input RPM.")
    parser.add_argument("--wf", type=float, default=250.0, help="Output RPM.")
    parser.add_argument("--power", type=float, default=8.0, help="Power HP.")
    return parser.parse_args()


def main():
    # Step 8: entry point dispatcher
    args = parse_args()
    if args.train:
        train(args.data)
    elif args.predict:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Model or scalers not found. Train first.")
        model = keras.models.load_model(MODEL_PATH, custom_objects={"stress_loss": stress_loss})
        scalers = load_scalers(SCALER_PATH)
        res = predict_single(model, scalers, args.wp, args.wf, args.power)
        for k, v in res.items():
            print(f"{k}: {v}")
    else:
        print("Specify --train or --predict")


if __name__ == "__main__":
    main()
