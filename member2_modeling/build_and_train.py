"""
Member 2 — Feature Engineering, Model Training & Artifact Export
Autonomous Multi-Agent EV Battery Life & Range Predictor

This script:
  1. Loads master_table.parquet (read-only, never modified)
  2. Extracts physics-based cycle-level features from raw telemetry
  3. Trains LightGBM and BiLSTM models using chronological train/val/test splits
  4. Evaluates against a persistence baseline
  5. Exports all artifacts for Member 4 integration

Chemistry support determined by data availability:
  - NMC (NASA): Full feature set including thermal features. SOH + RUL supported.
  - LFP (B5):  No thermal features. SOH supported (CONDITIONAL labels).
  - Na-ion:    Limited — shallow degradation, mixed charge/discharge, no temp.
               SOH estimation only (early-life). Excluded from primary models.
  - LFP B1:    Excluded (1 cycle, cannot train).
"""

import os
import sys
import json
import warnings
import hashlib
import datetime

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Configuration ───────────────────────────────────────────────────────────

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(WORKSPACE, "master_table.parquet")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Rated capacities per chemistry (from build script)
RATED_CAPACITY = {"NMC": 2.0, "LFP": 1.25, "Na-ion": 1.0}

# EOL threshold for RUL (diagnostic, only NMC supported)
EOL_THRESHOLD_PCT = 80.0

# Chronological split ratios
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20

CYCLE_KEY = ["dataset_id", "cell_id", "protocol_id", "cycle_id"]


# ─── 1. Feature Extraction ──────────────────────────────────────────────────

def extract_cycle_features(group, chemistry):
    """
    Extract physics-based features from a single cycle's telemetry.
    
    For NMC and LFP B5: telemetry is discharge-only.
    For Na-ion: we separate discharge phase (current_a < 0) before extracting.
    
    Returns a dict of feature values.
    """
    # For Na-ion, separate discharge phase by current sign
    if chemistry == "Na-ion":
        dis_mask = group["current_a"] < 0
        dis_group = group[dis_mask]
        chg_mask = group["current_a"] > 0
        chg_group = group[chg_mask]
        if len(dis_group) < 5:
            return None  # insufficient discharge data
    else:
        dis_group = group
        chg_group = pd.DataFrame()
    
    v = dis_group["voltage_v"].values
    i = dis_group["current_a"].values
    t = dis_group["time_sec"].values
    
    if len(v) < 5:
        return None
    
    # Time features
    dt = np.diff(t)
    dt = np.clip(dt, 0, None)  # ensure non-negative
    total_time_s = t[-1] - t[0] if len(t) > 1 else 0.0
    
    features = {}
    
    # ── Voltage statistics (discharge) ──
    features["v_mean"] = float(np.mean(v))
    features["v_std"] = float(np.std(v))
    features["v_min"] = float(np.min(v))
    features["v_max"] = float(np.max(v))
    features["v_range"] = float(np.max(v) - np.min(v))
    features["v_start"] = float(v[0])
    features["v_end"] = float(v[-1])
    features["v_delta"] = float(v[-1] - v[0])
    
    # Voltage percentiles
    features["v_q10"] = float(np.percentile(v, 10))
    features["v_q25"] = float(np.percentile(v, 25))
    features["v_q50"] = float(np.percentile(v, 50))
    features["v_q75"] = float(np.percentile(v, 75))
    features["v_q90"] = float(np.percentile(v, 90))
    
    # ── Current statistics (discharge) ──
    features["i_mean"] = float(np.mean(i))
    features["i_std"] = float(np.std(i))
    features["i_min"] = float(np.min(i))
    features["i_max"] = float(np.max(i))
    
    # ── Time features ──
    features["discharge_time_s"] = float(total_time_s)
    features["n_samples"] = int(len(v))
    
    # ── Energy & charge integrals (trapezoidal) ──
    if len(dt) > 0:
        # Ah throughput (integral of |I| * dt / 3600)
        ah_throughput = float(np.sum(np.abs(i[:-1]) * dt) / 3600.0)
        # Energy (integral of V * |I| * dt / 3600) in Wh
        energy_wh = float(np.sum(v[:-1] * np.abs(i[:-1]) * dt) / 3600.0)
    else:
        ah_throughput = 0.0
        energy_wh = 0.0
    
    features["ah_throughput"] = ah_throughput
    features["energy_wh"] = energy_wh
    features["avg_power_w"] = float(energy_wh * 3600 / total_time_s) if total_time_s > 0 else 0.0
    
    # ── Voltage curve shape features ──
    # Slope of voltage vs time (linear fit)
    if total_time_s > 0 and len(v) > 2:
        t_norm = (t - t[0]) / total_time_s
        try:
            slope, intercept = np.polyfit(t_norm, v, 1)
            features["v_slope"] = float(slope)
        except (np.linalg.LinAlgError, ValueError):
            features["v_slope"] = 0.0
    else:
        features["v_slope"] = 0.0
    
    # dV/dt statistics
    if len(dt) > 0:
        dv = np.diff(v)
        dt_safe = np.where(dt > 0, dt, 1.0)
        dvdt = dv / dt_safe
        features["dvdt_mean"] = float(np.mean(dvdt))
        features["dvdt_std"] = float(np.std(dvdt))
        features["dvdt_min"] = float(np.min(dvdt))
        features["dvdt_max"] = float(np.max(dvdt))
    else:
        features["dvdt_mean"] = 0.0
        features["dvdt_std"] = 0.0
        features["dvdt_min"] = 0.0
        features["dvdt_max"] = 0.0
    
    # ── Area under voltage curve (V-t integral) ──
    if len(dt) > 0:
        v_integral = float(np.sum(v[:-1] * dt))
        features["v_time_integral"] = v_integral
    else:
        features["v_time_integral"] = 0.0
    
    # ── Thermal features (NMC only — temp_c present) ──
    if chemistry == "NMC" and "temp_c" in dis_group.columns:
        temp = dis_group["temp_c"].dropna().values
        if len(temp) > 0:
            features["temp_mean"] = float(np.mean(temp))
            features["temp_std"] = float(np.std(temp))
            features["temp_min"] = float(np.min(temp))
            features["temp_max"] = float(np.max(temp))
            features["temp_range"] = float(np.max(temp) - np.min(temp))
            features["temp_start"] = float(temp[0])
            features["temp_end"] = float(temp[-1])
            features["temp_rise"] = float(temp[-1] - temp[0])
        else:
            for k in ["temp_mean", "temp_std", "temp_min", "temp_max",
                       "temp_range", "temp_start", "temp_end", "temp_rise"]:
                features[k] = np.nan
    
    return features


def build_cycle_feature_table(df):
    """
    Build a cycle-level feature table from the telemetry DataFrame.
    Returns (features_df, labels_df) with matching indices.
    """
    print("  Extracting cycle-level features...")
    
    # Extract labels first (one per cycle)
    label_df = df.groupby(CYCLE_KEY, sort=False).agg(
        capacity_ah=("capacity_ah", "first"),
        soh_pct=("soh_pct", "first"),
        chemistry=("chemistry", "first"),
    ).reset_index()
    
    # Drop label columns from telemetry for feature extraction
    telem_df = df.drop(columns=["capacity_ah", "soh_pct", "quality_flag", "source_file"])
    
    all_features = []
    for keys, group in telem_df.groupby(CYCLE_KEY, sort=False):
        chem = group["chemistry"].iloc[0]
        feats = extract_cycle_features(group, chem)
        if feats is None:
            continue
        # Add identifying columns
        feats["dataset_id"] = keys[0]
        feats["cell_id"] = keys[1]
        feats["protocol_id"] = keys[2]
        feats["cycle_id"] = keys[3]
        all_features.append(feats)
    
    feat_df = pd.DataFrame(all_features)
    print(f"  Extracted features for {len(feat_df)} cycles")
    
    # Merge with labels
    merged = feat_df.merge(label_df, on=CYCLE_KEY, how="inner")
    
    return merged


# ─── 2. Chronological Splitting ─────────────────────────────────────────────

def chronological_split(cell_df):
    """
    Split a single cell's data chronologically by positional index
    in the sorted cycle list (60/20/20).
    """
    cell_df = cell_df.sort_values("cycle_id").reset_index(drop=True)
    n = len(cell_df)
    n_train = int(np.floor(n * TRAIN_FRAC))
    n_val = int(np.floor(n * VAL_FRAC))
    
    train = cell_df.iloc[:n_train].copy()
    val = cell_df.iloc[n_train:n_train + n_val].copy()
    test = cell_df.iloc[n_train + n_val:].copy()
    
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    
    return pd.concat([train, val, test], ignore_index=True)


# ─── 3. Task C: Future-Cycle SOH Prediction ─────────────────────────────────

def create_future_soh_dataset(cell_df, horizon=1):
    """
    For Task C: create input-output pairs where input is a cycle's features
    and target is the SOH of the next observed record.

    Uses positional indexing (not cycle_id arithmetic), so this is an
    observed-record horizon, not a guaranteed physical-cycle horizon.
    For LFP B5 the observed transitions 100->151 and 200->401 cross large
    missing-ID gaps and must not be described as one physical cycle apart.
    """
    cell_df = cell_df.sort_values("cycle_id").reset_index(drop=True)
    
    if len(cell_df) <= horizon:
        return pd.DataFrame()
    
    inputs = cell_df.iloc[:len(cell_df) - horizon].copy()
    targets = cell_df.iloc[horizon:].copy()
    
    # The target is the future SOH
    inputs = inputs.reset_index(drop=True)
    targets = targets.reset_index(drop=True)
    inputs["target_soh_pct"] = targets["soh_pct"].values
    inputs["target_cycle_id"] = targets["cycle_id"].values
    
    # Also add current SOH as a feature (legitimate for Task C since 
    # we're predicting FUTURE SOH from current + past observations)
    inputs["current_soh_pct"] = inputs["soh_pct"]
    
    return inputs


# ─── 4. RUL Label Construction (NMC only) ────────────────────────────────────

def compute_rul_labels(cell_df, eol_threshold=EOL_THRESHOLD_PCT):
    """
    For NMC cells that cross the EOL threshold, compute RUL as the
    number of observed cycles remaining until first crossing.
    Returns the cell_df with an additional 'rul_cycles' column.
    """
    cell_df = cell_df.sort_values("cycle_id").reset_index(drop=True)
    
    # Find first crossing position
    below_eol = cell_df["soh_pct"] < eol_threshold
    if not below_eol.any():
        cell_df["rul_cycles"] = np.nan
        return cell_df
    
    first_crossing_pos = below_eol.idxmax()
    
    # RUL = distance (in positions) to first crossing
    cell_df["rul_cycles"] = np.nan
    for pos in range(len(cell_df)):
        if pos <= first_crossing_pos:
            cell_df.loc[pos, "rul_cycles"] = float(first_crossing_pos - pos)
        # After EOL, RUL = 0
        else:
            cell_df.loc[pos, "rul_cycles"] = 0.0
    
    return cell_df


# ─── 5. Model Training ──────────────────────────────────────────────────────

def get_feature_columns(df, task="soh"):
    """
    Return the ordered list of feature columns based on what's available.
    Excludes identifying columns, labels, and split markers.
    """
    exclude = {"dataset_id", "cell_id", "protocol_id", "cycle_id",
               "capacity_ah", "soh_pct", "chemistry", "split",
               "target_soh_pct", "target_cycle_id", "current_soh_pct",
               "rul_cycles", "cycle_position"}
    
    # Include current_soh_pct as a feature for Task C
    if task == "future_soh":
        exclude.discard("current_soh_pct")
    
    cols = [c for c in df.columns if c not in exclude]
    # Drop columns with all NaN
    valid_cols = [c for c in cols if df[c].notna().any()]
    return sorted(valid_cols)


def train_lightgbm(X_train, y_train, X_val, y_val, task_name="soh"):
    """Train a LightGBM regressor."""
    import lightgbm as lgb
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "seed": 42,
        "n_estimators": 500,
    }
    
    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    
    print(f"    LightGBM ({task_name}): best_iteration={model.best_iteration_}")
    return model


import torch
import torch.nn as nn

class BiLSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last time step
        out = lstm_out[:, -1, :]
        return self.fc(out).squeeze(-1)


def train_bilstm(X_train_seq, y_train, X_val_seq, y_val, input_dim, task_name="soh"):
    """
    Train a BiLSTM model for sequence-based SOH prediction.
    X_train_seq is expected to be (N, seq_len, n_features).
    """
    from torch.utils.data import DataLoader, TensorDataset

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_tr = torch.FloatTensor(X_train_seq).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_vl = torch.FloatTensor(X_val_seq).to(device)
    y_vl = torch.FloatTensor(y_val).to(device)
    
    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    
    model = BiLSTMPredictor(input_dim, hidden_dim=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    
    for epoch in range(200):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl)
            val_loss = criterion(val_pred, y_vl).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 30:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"    BiLSTM ({task_name}): best_val_rmse={np.sqrt(best_val_loss):.4f}, epochs={epoch+1}")
    
    return model, device


def create_sequences(df, feature_cols, target_col, seq_len=5):
    """
    Create sliding-window sequences for BiLSTM from a per-cell sorted DataFrame.
    Returns (X_seq, y_seq) arrays.
    """
    X_all, y_all = [], []
    
    for cell_id, cell_df in df.groupby("cell_id"):
        cell_df = cell_df.sort_values("cycle_id").reset_index(drop=True)
        X_vals = cell_df[feature_cols].values.astype(np.float32)
        y_vals = cell_df[target_col].values.astype(np.float32)
        
        for i in range(len(cell_df) - seq_len + 1):
            X_all.append(X_vals[i:i + seq_len])
            y_all.append(y_vals[i + seq_len - 1])
    
    if len(X_all) == 0:
        return np.array([]).reshape(0, seq_len, len(feature_cols)), np.array([])
    
    return np.array(X_all), np.array(y_all)


# ─── 6. Evaluation ──────────────────────────────────────────────────────────

def evaluate_predictions(y_true, y_pred, label=""):
    """Compute regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    if len(y_true) == 0:
        return {"label": label, "n": 0, "mae": None, "rmse": None, "r2": None, "mape": None}
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    # MAPE (avoid division by zero)
    nonzero = np.abs(y_true) > 1e-6
    if nonzero.any():
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = None
    
    return {"label": label, "n": int(len(y_true)), "mae": mae, "rmse": rmse, "r2": r2, "mape_pct": mape}


def persistence_baseline(train_df, test_df, target_col="soh_pct"):
    """
    Persistence baseline: predict soh(N) = soh(N-1).
    For future-SOH task: predict target_soh = current_soh.
    """
    if target_col == "target_soh_pct" and "current_soh_pct" in test_df.columns:
        return test_df["current_soh_pct"].values
    else:
        return test_df["soh_pct"].values  # naive: no degradation


# ─── 7. Main Pipeline ───────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MEMBER 2: Feature Engineering & Model Training Pipeline")
    print("=" * 70)
    
    # ── Load data ──
    print("\n[1/7] Loading master_table.parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Loaded {len(df)} rows, {df['chemistry'].nunique()} chemistries")
    
    # Verify parquet hash
    with open(PARQUET_PATH, "rb") as f:
        pq_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"  SHA256: {pq_hash}")
    
    # ── Exclude B1 (1 cycle) ──
    df = df[df["dataset_id"] != "LFP_MultiStat_B1"].copy()
    print(f"  After excluding B1: {len(df)} rows")
    
    # ── Extract features ──
    print("\n[2/7] Extracting cycle-level features...")
    cycle_df = build_cycle_feature_table(df)
    print(f"  Total cycles with features: {len(cycle_df)}")
    
    # ── Drop cycles with missing SOH ──
    cycle_df = cycle_df.dropna(subset=["soh_pct"]).copy()
    print(f"  After dropping null SOH: {len(cycle_df)}")
    
    # ── Add positional cycle index per cell ──
    cycle_df = cycle_df.sort_values(CYCLE_KEY)
    cycle_df["cycle_position"] = cycle_df.groupby("cell_id").cumcount()
    
    # ── Report cycle counts ──
    for chem, chem_df in cycle_df.groupby("chemistry"):
        cells = chem_df["cell_id"].unique()
        print(f"  {chem}: {len(chem_df)} cycles across {len(cells)} cells")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TASK C: Future-Cycle SOH Prediction (Primary Task)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[3/7] Building Task C: Future-Cycle SOH datasets...")
    
    # We focus on NMC (strong degradation signal, thermal features, 
    # validated labels, RUL capability) and LFP B5 (moderate degradation,
    # CONDITIONAL labels, large cycle count).
    
    # ── NMC dataset ──
    nmc_df = cycle_df[cycle_df["chemistry"] == "NMC"].copy()
    
    # Build future-SOH dataset (horizon=1) per cell
    nmc_future_dfs = []
    for cell_id, cell_data in nmc_df.groupby("cell_id"):
        fut = create_future_soh_dataset(cell_data, horizon=1)
        if len(fut) > 0:
            nmc_future_dfs.append(fut)
    nmc_future = pd.concat(nmc_future_dfs, ignore_index=True) if nmc_future_dfs else pd.DataFrame()
    print(f"  NMC future-SOH samples: {len(nmc_future)}")
    
    # ── LFP B5 dataset ──
    lfp_df = cycle_df[cycle_df["dataset_id"] == "LFP_MultiStat_B5"].copy()
    lfp_future_dfs = []
    for cell_id, cell_data in lfp_df.groupby("cell_id"):
        fut = create_future_soh_dataset(cell_data, horizon=1)
        if len(fut) > 0:
            lfp_future_dfs.append(fut)
    lfp_future = pd.concat(lfp_future_dfs, ignore_index=True) if lfp_future_dfs else pd.DataFrame()
    print(f"  LFP B5 future-SOH samples: {len(lfp_future)}")
    
    # ── Combined NMC + LFP for a cross-chemistry model ──
    # Thermal columns will be NaN for LFP — we handle them by training
    # separate NMC-only and combined models.
    
    # ═══════════════════════════════════════════════════════════════════════
    # Chronological splits (per cell)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[4/7] Applying chronological splits...")
    
    # NMC splits
    nmc_split_dfs = []
    for cell_id, cell_data in nmc_future.groupby("cell_id"):
        nmc_split_dfs.append(chronological_split(cell_data))
    nmc_split = pd.concat(nmc_split_dfs, ignore_index=True)
    
    for s in ["train", "val", "test"]:
        n = (nmc_split["split"] == s).sum()
        print(f"  NMC {s}: {n} samples")
    
    # LFP splits  
    lfp_split_dfs = []
    for cell_id, cell_data in lfp_future.groupby("cell_id"):
        lfp_split_dfs.append(chronological_split(cell_data))
    lfp_split = pd.concat(lfp_split_dfs, ignore_index=True)
    
    for s in ["train", "val", "test"]:
        n = (lfp_split["split"] == s).sum()
        print(f"  LFP {s}: {n} samples")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Determine feature columns
    # ═══════════════════════════════════════════════════════════════════════
    
    # NMC features (includes thermal)
    nmc_feature_cols = get_feature_columns(nmc_split, task="future_soh")
    print(f"\n  NMC feature columns ({len(nmc_feature_cols)}): {nmc_feature_cols}")
    
    # LFP features (no thermal — drop temp_* columns if present)
    lfp_feature_cols = [c for c in get_feature_columns(lfp_split, task="future_soh")
                        if not c.startswith("temp_")]
    print(f"  LFP feature columns ({len(lfp_feature_cols)}): {lfp_feature_cols}")
    
    # Common features (intersection, for potential cross-chemistry model)
    common_feature_cols = sorted(set(nmc_feature_cols) & set(lfp_feature_cols))
    print(f"  Common feature columns ({len(common_feature_cols)}): {common_feature_cols}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Scale features
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[5/7] Scaling features and training models...")
    
    from sklearn.preprocessing import StandardScaler
    
    results = {}  # Collect all evaluation results
    models = {}   # Collect trained models
    scalers = {}  # Collect fitted scalers
    
    # ── Model 1: NMC LightGBM (Task C - Future SOH) ──
    print("\n  --- NMC LightGBM: Future-Cycle SOH Prediction ---")
    
    nmc_train = nmc_split[nmc_split["split"] == "train"]
    nmc_val = nmc_split[nmc_split["split"] == "val"]
    nmc_test = nmc_split[nmc_split["split"] == "test"]
    
    # Fill NaN in thermal features with column median from training set
    for col in nmc_feature_cols:
        if nmc_train[col].isna().any():
            median_val = nmc_train[col].median()
            nmc_train = nmc_train.copy()
            nmc_train[col] = nmc_train[col].fillna(median_val)
            nmc_val = nmc_val.copy()
            nmc_val[col] = nmc_val[col].fillna(median_val)
            nmc_test = nmc_test.copy()
            nmc_test[col] = nmc_test[col].fillna(median_val)
    
    scaler_nmc = StandardScaler()
    X_train_nmc = scaler_nmc.fit_transform(nmc_train[nmc_feature_cols].values)
    X_val_nmc = scaler_nmc.transform(nmc_val[nmc_feature_cols].values)
    X_test_nmc = scaler_nmc.transform(nmc_test[nmc_feature_cols].values)
    
    y_train_nmc = nmc_train["target_soh_pct"].values
    y_val_nmc = nmc_val["target_soh_pct"].values
    y_test_nmc = nmc_test["target_soh_pct"].values
    
    # Train LightGBM (uses unscaled features — LightGBM is tree-based)
    lgbm_nmc = train_lightgbm(
        nmc_train[nmc_feature_cols].values, y_train_nmc,
        nmc_val[nmc_feature_cols].values, y_val_nmc,
        task_name="nmc_future_soh"
    )
    
    pred_lgbm_nmc_test = lgbm_nmc.predict(nmc_test[nmc_feature_cols].values)
    pred_lgbm_nmc_val = lgbm_nmc.predict(nmc_val[nmc_feature_cols].values)
    
    results["nmc_lgbm_test"] = evaluate_predictions(y_test_nmc, pred_lgbm_nmc_test, "NMC LightGBM Test")
    results["nmc_lgbm_val"] = evaluate_predictions(y_val_nmc, pred_lgbm_nmc_val, "NMC LightGBM Val")
    
    # Persistence baseline
    pred_persist_nmc = persistence_baseline(nmc_train, nmc_test, "target_soh_pct")
    results["nmc_persistence_test"] = evaluate_predictions(y_test_nmc, pred_persist_nmc, "NMC Persistence Test")
    
    models["nmc_lgbm"] = lgbm_nmc
    scalers["nmc"] = scaler_nmc
    
    print(f"    NMC LightGBM Test: {results['nmc_lgbm_test']}")
    print(f"    NMC Persistence Test: {results['nmc_persistence_test']}")
    
    # ── Model 2: LFP B5 LightGBM (Task C - Future SOH) ──
    print("\n  --- LFP B5 LightGBM: Future-Cycle SOH Prediction ---")
    
    lfp_train = lfp_split[lfp_split["split"] == "train"]
    lfp_val = lfp_split[lfp_split["split"] == "val"]
    lfp_test = lfp_split[lfp_split["split"] == "test"]
    
    scaler_lfp = StandardScaler()
    X_train_lfp = scaler_lfp.fit_transform(lfp_train[lfp_feature_cols].values)
    X_val_lfp = scaler_lfp.transform(lfp_val[lfp_feature_cols].values)
    X_test_lfp = scaler_lfp.transform(lfp_test[lfp_feature_cols].values)
    
    y_train_lfp = lfp_train["target_soh_pct"].values
    y_val_lfp = lfp_val["target_soh_pct"].values
    y_test_lfp = lfp_test["target_soh_pct"].values
    
    lgbm_lfp = train_lightgbm(
        lfp_train[lfp_feature_cols].values, y_train_lfp,
        lfp_val[lfp_feature_cols].values, y_val_lfp,
        task_name="lfp_future_soh"
    )
    
    pred_lgbm_lfp_test = lgbm_lfp.predict(lfp_test[lfp_feature_cols].values)
    pred_lgbm_lfp_val = lgbm_lfp.predict(lfp_val[lfp_feature_cols].values)
    
    results["lfp_lgbm_test"] = evaluate_predictions(y_test_lfp, pred_lgbm_lfp_test, "LFP LightGBM Test")
    results["lfp_lgbm_val"] = evaluate_predictions(y_val_lfp, pred_lgbm_lfp_val, "LFP LightGBM Val")
    
    pred_persist_lfp = persistence_baseline(lfp_train, lfp_test, "target_soh_pct")
    results["lfp_persistence_test"] = evaluate_predictions(y_test_lfp, pred_persist_lfp, "LFP Persistence Test")
    
    models["lfp_lgbm"] = lgbm_lfp
    scalers["lfp"] = scaler_lfp
    
    print(f"    LFP LightGBM Test: {results['lfp_lgbm_test']}")
    print(f"    LFP Persistence Test: {results['lfp_persistence_test']}")
    
    # ── Model 3: NMC BiLSTM (Task C - Future SOH, sequence-based) ──
    print("\n  --- NMC BiLSTM: Future-Cycle SOH Prediction (Sequence) ---")
    
    SEQ_LEN = 5
    
    # Build sequences per cell from chronological splits
    # For BiLSTM we use sequences of features to predict future SOH
    bilstm_feature_cols = nmc_feature_cols.copy()
    scaler_nmc_seq = scaler_nmc
    
    def make_cell_sequences(split_df, feature_cols, scaler, target_col="target_soh_pct", seq_len=5):
        X_out, y_out = [], []
        for cell_id, cdf in split_df.groupby("cell_id"):
            cdf = cdf.sort_values("cycle_id").reset_index(drop=True)
            if len(cdf) < seq_len:
                continue
            X_vals = scaler.transform(cdf[feature_cols].values).astype(np.float32)
            y_vals = cdf[target_col].values.astype(np.float32)
            for i in range(len(cdf) - seq_len + 1):
                X_out.append(X_vals[i:i + seq_len])
                y_out.append(y_vals[i + seq_len - 1])
        if not X_out:
            return np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return np.array(X_out, dtype=np.float32), np.array(y_out, dtype=np.float32)
    
    X_train_seq, y_train_seq = make_cell_sequences(nmc_train, bilstm_feature_cols, scaler_nmc_seq, "target_soh_pct", SEQ_LEN)
    X_val_seq, y_val_seq = make_cell_sequences(nmc_val, bilstm_feature_cols, scaler_nmc_seq, "target_soh_pct", SEQ_LEN)
    X_test_seq, y_test_seq = make_cell_sequences(nmc_test, bilstm_feature_cols, scaler_nmc_seq, "target_soh_pct", SEQ_LEN)
    
    print(f"    Sequence shapes: train={X_train_seq.shape}, val={X_val_seq.shape}, test={X_test_seq.shape}")
    
    bilstm_model = None
    if X_train_seq.shape[0] >= 10 and X_val_seq.shape[0] >= 3:
        bilstm_model, bilstm_device = train_bilstm(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            input_dim=len(bilstm_feature_cols),
            task_name="nmc_soh_seq"
        )
        
        import torch
        bilstm_model.eval()
        with torch.no_grad():
            pred_bilstm_test = bilstm_model(
                torch.FloatTensor(X_test_seq).to(bilstm_device)
            ).cpu().numpy()
            pred_bilstm_val = bilstm_model(
                torch.FloatTensor(X_val_seq).to(bilstm_device)
            ).cpu().numpy()
        
        results["nmc_bilstm_test"] = evaluate_predictions(y_test_seq, pred_bilstm_test, "NMC BiLSTM Test")
        results["nmc_bilstm_val"] = evaluate_predictions(y_val_seq, pred_bilstm_val, "NMC BiLSTM Val")
        print(f"    NMC BiLSTM Test: {results['nmc_bilstm_test']}")
        
        models["nmc_bilstm"] = bilstm_model
        scalers["nmc_seq"] = scaler_nmc_seq
    else:
        print("    WARN: Insufficient sequences for BiLSTM training")
    
    # ═══════════════════════════════════════════════════════════════════════
    # RUL Model (NMC only)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  --- NMC LightGBM: RUL Estimation ---")
    
    nmc_rul_dfs = []
    for cell_id, cell_data in nmc_df.groupby("cell_id"):
        rul_data = compute_rul_labels(cell_data.copy())
        nmc_rul_dfs.append(rul_data)
    nmc_rul = pd.concat(nmc_rul_dfs, ignore_index=True)
    nmc_rul["current_soh_pct"] = nmc_rul["soh_pct"]
    
    # Only keep cycles with valid RUL (cells that cross EOL)
    nmc_rul = nmc_rul.dropna(subset=["rul_cycles"])
    print(f"    NMC cycles with RUL labels: {len(nmc_rul)}")
    
    if len(nmc_rul) > 10:
        # Chronological split
        rul_split_dfs = []
        for cell_id, cell_data in nmc_rul.groupby("cell_id"):
            rul_split_dfs.append(chronological_split(cell_data))
        rul_split = pd.concat(rul_split_dfs, ignore_index=True)
        
        rul_train = rul_split[rul_split["split"] == "train"]
        rul_val = rul_split[rul_split["split"] == "val"]
        rul_test = rul_split[rul_split["split"] == "test"]
        
        rul_feature_cols = nmc_feature_cols.copy()
        
        # Fill NaN
        for col in rul_feature_cols:
            med = rul_train[col].median()
            rul_train = rul_train.copy()
            rul_train[col] = rul_train[col].fillna(med)
            rul_val = rul_val.copy()
            rul_val[col] = rul_val[col].fillna(med)
            rul_test = rul_test.copy()
            rul_test[col] = rul_test[col].fillna(med)
        
        lgbm_rul = train_lightgbm(
            rul_train[rul_feature_cols].values, rul_train["rul_cycles"].values,
            rul_val[rul_feature_cols].values, rul_val["rul_cycles"].values,
            task_name="nmc_rul"
        )
        
        pred_rul_test = lgbm_rul.predict(rul_test[rul_feature_cols].values)
        results["nmc_rul_test"] = evaluate_predictions(
            rul_test["rul_cycles"].values, pred_rul_test, "NMC RUL Test"
        )
        print(f"    NMC RUL Test: {results['nmc_rul_test']}")
        
        models["nmc_rul_lgbm"] = lgbm_rul
    
    # ═══════════════════════════════════════════════════════════════════════
    # [6/7] Export Artifacts
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[6/7] Exporting model artifacts...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save LightGBM models
    for name, model in models.items():
        if "bilstm" not in name:
            path = os.path.join(OUTPUT_DIR, f"{name}_model.joblib")
            joblib.dump(model, path)
            print(f"  Saved {path}")
    
    # Save BiLSTM model
    if "nmc_bilstm" in models:
        import torch
        bilstm_path = os.path.join(OUTPUT_DIR, "nmc_bilstm_model.pt")
        torch.save(models["nmc_bilstm"].state_dict(), bilstm_path)
        print(f"  Saved {bilstm_path}")
    
    # Save scalers
    for name, scaler in scalers.items():
        path = os.path.join(OUTPUT_DIR, f"scaler_{name}.joblib")
        joblib.dump(scaler, path)
        print(f"  Saved {path}")
    
    # ── Feature Schema ──
    feature_schema = {
        "schema_version": "1.0.0",
        "created": datetime.datetime.now().isoformat(),
        "parquet_sha256": pq_hash,
        "task": "future_cycle_soh_prediction",
        "prediction_horizon_cycles": 1,
        "prediction_horizon_semantics": (
            "Positional shift by 1 observed cycle record. The target is the SOH "
            "of the next observed record, not necessarily the next physical "
            "cycle. For B5, 100->151 and 200->401 cross missing-ID gaps, and "
            "39->41 follows the dropped null-label cycle 40."
        ),
        "models": {
            "nmc_lgbm": {
                "chemistry": "NMC",
                "model_status": "experimental_research_stage",
                "feature_names": nmc_feature_cols,
                "feature_count": len(nmc_feature_cols),
                "target": "soh_pct of next observed cycle record (positional shift; not necessarily next physical cycle)",
                "target_unit": "%",
                "model_file": "nmc_lgbm_model.joblib",
                "scaler_file": "scaler_nmc.joblib",
                "scaler_required": False,  # LightGBM is tree-based
                "supports_rul": False,
            },
            "lfp_lgbm": {
                "chemistry": "LFP",
                "model_status": "experimental_research_stage_conditional_labels",
                "feature_names": lfp_feature_cols,
                "feature_count": len(lfp_feature_cols),
                "target": "soh_pct of next observed cycle record (positional shift; not necessarily next physical cycle)",
                "target_unit": "%",
                "model_file": "lfp_lgbm_model.joblib",
                "scaler_file": "scaler_lfp.joblib",
                "scaler_required": False,
                "supports_rul": False,
                "caveats": [
                    "B5 SOH labels are CONDITIONAL due to charge-balance discrepancy (36.3% of transitions increase)",
                    "B5 observed IDs form blocks 1-100, 151-200, 401-500; IDs 101-150 and 201-400 are absent",
                    "cycle_id=40 has a null SOH label and is dropped; 249 labelled cycles yield 248 model-ready pairs",
                    "Learned SOH model underperforms the persistence baseline (test MAE 0.9256 vs 0.1313, n=51)",
                ],
            },
            "nmc_rul_lgbm": {
                "chemistry": "NMC",
                "model_status": "unvalidated_pre_eol",
                "feature_names": nmc_feature_cols if "nmc_rul_lgbm" in models else [],
                "feature_count": len(nmc_feature_cols) if "nmc_rul_lgbm" in models else 0,
                "target": "rul_cycles (remaining observed cycles to 80% EOL; NMC cycle IDs are contiguous)",
                "target_unit": "cycles",
                "eol_threshold_pct": EOL_THRESHOLD_PCT,
                "model_file": "nmc_rul_lgbm_model.joblib" if "nmc_rul_lgbm" in models else None,
                "scaler_file": "scaler_nmc.joblib",
                "scaler_required": False,
            },
        },
        "feature_definitions": {},
        "blocked_features": {
            "mean_temp_c": "BLOCKED — temp_c is NULL for all non-NMC rows. Available only for NMC chemistry.",
            "max_c_rate": "BLOCKED — current_a units are UNDERDETERMINED for LFP. C-rate requires validated (current_a / rated_capacity) with confirmed unit. Not derivable without independent calibration.",
            "target_soc": "BLOCKED — no SOC column exists in master_table.parquet. SOC requires cumulative coulomb counting or OCV-SOC lookup table, neither available.",
            "min_soc": "BLOCKED — same as target_soc.",
            "hours_above_80soc_per_cycle": "BLOCKED — requires SOC time-series, which is not available.",
            "fast_charge_fraction": "BLOCKED — requires charge-phase telemetry with validated C-rate thresholds. NASA NMC and LFP B5 contain discharge-only telemetry. Na-ion has mixed phases but current units need validation.",
            "n_cycles": "SUPPORTED as cycle_position — available as positional index in the observed cycle list per cell.",
        },
    }
    
    # Build feature definitions
    feat_def_template = {
        "v_mean": {"unit": "V", "description": "Mean discharge voltage", "dtype": "float64"},
        "v_std": {"unit": "V", "description": "Std dev of discharge voltage", "dtype": "float64"},
        "v_min": {"unit": "V", "description": "Minimum discharge voltage", "dtype": "float64"},
        "v_max": {"unit": "V", "description": "Maximum discharge voltage", "dtype": "float64"},
        "v_range": {"unit": "V", "description": "Voltage range (max - min)", "dtype": "float64"},
        "v_start": {"unit": "V", "description": "Starting voltage of discharge", "dtype": "float64"},
        "v_end": {"unit": "V", "description": "Ending voltage of discharge", "dtype": "float64"},
        "v_delta": {"unit": "V", "description": "Voltage change (end - start)", "dtype": "float64"},
        "v_q10": {"unit": "V", "description": "10th percentile discharge voltage", "dtype": "float64"},
        "v_q25": {"unit": "V", "description": "25th percentile discharge voltage", "dtype": "float64"},
        "v_q50": {"unit": "V", "description": "Median discharge voltage", "dtype": "float64"},
        "v_q75": {"unit": "V", "description": "75th percentile discharge voltage", "dtype": "float64"},
        "v_q90": {"unit": "V", "description": "90th percentile discharge voltage", "dtype": "float64"},
        "i_mean": {"unit": "A", "description": "Mean discharge current", "dtype": "float64"},
        "i_std": {"unit": "A", "description": "Std dev of discharge current", "dtype": "float64"},
        "i_min": {"unit": "A", "description": "Min discharge current", "dtype": "float64"},
        "i_max": {"unit": "A", "description": "Max discharge current", "dtype": "float64"},
        "discharge_time_s": {"unit": "s", "description": "Total discharge duration", "dtype": "float64"},
        "n_samples": {"unit": "count", "description": "Number of telemetry samples in discharge", "dtype": "int64"},
        "ah_throughput": {"unit": "Ah", "description": "Ah throughput (integral |I|*dt)", "dtype": "float64"},
        "energy_wh": {"unit": "Wh", "description": "Discharge energy (integral V*|I|*dt)", "dtype": "float64"},
        "avg_power_w": {"unit": "W", "description": "Average discharge power", "dtype": "float64"},
        "v_slope": {"unit": "V/normalized_t", "description": "Linear voltage slope over normalized time", "dtype": "float64"},
        "dvdt_mean": {"unit": "V/s", "description": "Mean dV/dt", "dtype": "float64"},
        "dvdt_std": {"unit": "V/s", "description": "Std dev of dV/dt", "dtype": "float64"},
        "dvdt_min": {"unit": "V/s", "description": "Min dV/dt", "dtype": "float64"},
        "dvdt_max": {"unit": "V/s", "description": "Max dV/dt", "dtype": "float64"},
        "v_time_integral": {"unit": "V·s", "description": "Area under voltage-time curve", "dtype": "float64"},
        "current_soh_pct": {"unit": "%", "description": "Current cycle SOH (input for Task C future prediction)", "dtype": "float64"},
        "temp_mean": {"unit": "°C", "description": "Mean temperature (NMC only)", "dtype": "float64"},
        "temp_std": {"unit": "°C", "description": "Std dev temperature (NMC only)", "dtype": "float64"},
        "temp_min": {"unit": "°C", "description": "Min temperature (NMC only)", "dtype": "float64"},
        "temp_max": {"unit": "°C", "description": "Max temperature (NMC only)", "dtype": "float64"},
        "temp_range": {"unit": "°C", "description": "Temperature range (NMC only)", "dtype": "float64"},
        "temp_start": {"unit": "°C", "description": "Start temperature (NMC only)", "dtype": "float64"},
        "temp_end": {"unit": "°C", "description": "End temperature (NMC only)", "dtype": "float64"},
        "temp_rise": {"unit": "°C", "description": "Temperature rise during discharge (NMC only)", "dtype": "float64"},
    }
    feature_schema["feature_definitions"] = feat_def_template
    
    schema_path = os.path.join(OUTPUT_DIR, "feature_schema.json")
    with open(schema_path, "w") as f:
        json.dump(feature_schema, f, indent=2)
    print(f"  Saved {schema_path}")
    
    # ── Model Manifest ──
    import lightgbm
    import sklearn
    
    torch_version = "N/A"
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        pass
    
    b5_observed_cycles = int(
        df.loc[df["dataset_id"] == "LFP_MultiStat_B5", "cycle_id"].nunique()
    )

    manifest = {
        "project": "Autonomous Multi-Agent EV Battery Life & Range Predictor",
        "module": "Member 2 — SOH Prediction & RUL Estimation",
        "created": datetime.datetime.now().isoformat(),
        "parquet_sha256": pq_hash,
        "prediction_horizon_cycles": 1,
        "prediction_horizon_semantics": (
            "Positional shift by 1 observed cycle record. The target is the SOH "
            "of the next observed record, not necessarily the next physical "
            "cycle. For B5, observed transitions include 100->151 (difference 51) "
            "and 200->401 (difference 201); the label filter also creates 39->41 "
            "(difference 2) because cycle_id=40 has a null SOH label."
        ),
        "eol_threshold_pct": EOL_THRESHOLD_PCT,
        "chronological_split": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "evaluation_results": results,
        "supported_chemistries": {
            "NMC": {
                "soh_prediction": True,
                "rul_estimation": "UNVALIDATED (Post-EOL test only)",
                "thermal_features": True,
                "status": "experimental_research_stage",
                "cells_trained_on": ["B0005", "B0006", "B0018"],
                "total_cycles": int(len(nmc_df)),
                "model_ready_pairs": int(len(nmc_future)),
            },
            "LFP": {
                "soh_prediction": True,
                "rul_estimation": False,
                "thermal_features": False,
                "status": "experimental_research_stage_conditional_labels",
                "cells_trained_on": ["BATTERY_5"],
                "total_cycles": b5_observed_cycles,
                "labeled_cycles": int(len(lfp_df)),
                "model_ready_pairs": int(len(lfp_future)),
                "caveats": [
                    "250 observed cycle IDs in cell B5 form three contiguous blocks: 1-100, 151-200 and 401-500. IDs 101-150 (50 IDs) and 201-400 (200 IDs) are absent, i.e. 250 missing IDs between the observed minimum and maximum.",
                    "cycle_id=40 has a null SOH label (625 rows). The pipeline drops it, leaving 249 labelled cycles and 248 model-ready input->target pairs; this creates a 39->41 gap-crossing pair.",
                    "The two raw structural transitions 100->151 (difference 51) and 200->401 (difference 201) survive the filters. Targets are the next observed record, not necessarily the next physical cycle.",
                    "SOH labels CONDITIONAL (documented charge-balance discrepancy; 36.3% SOH transitions increase)",
                    "Learned SOH model underperforms persistence baseline: test MAE 0.9256 vs 0.1313 (n=51, adjacent observed pairs only)",
                    "B1 excluded (1 cycle)",
                ],
            },
            "Na-ion": {
                "soh_prediction": False,
                "rul_estimation": False,
                "thermal_features": False,
                "reason": "Insufficient degradation (<1.1% fade over 25 cycles). Mixed charge/discharge telemetry. Excluded from model training.",
            },
        },
        "files": {
            "nmc_lgbm_model.joblib": "LightGBM model for NMC future-cycle SOH prediction",
            "lfp_lgbm_model.joblib": "LightGBM model for LFP future-cycle SOH prediction",
            "nmc_rul_lgbm_model.joblib": "LightGBM model for NMC RUL estimation (if trained)",
            "nmc_bilstm_model.pt": "PyTorch BiLSTM model for NMC SOH sequence prediction",
            "scaler_nmc.joblib": "StandardScaler fitted on NMC training features",
            "scaler_lfp.joblib": "StandardScaler fitted on LFP training features",
            "scaler_nmc_seq.joblib": "StandardScaler for BiLSTM sequence features",
            "feature_schema.json": "Complete feature schema with definitions and blocked features",
            "model_manifest.json": "This file — metadata, metrics, and integration instructions",
            "predictor.py": "Python inference API for Member 4 integration",
        },
        "dependencies": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lightgbm.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "torch": torch_version,
            "pyarrow": None,  # filled below
        },
    }
    
    try:
        import pyarrow
        manifest["dependencies"]["pyarrow"] = pyarrow.__version__
    except ImportError:
        pass

    # ── Model Readiness (status labels, not production claims) ──
    def _beats_persistence(model_key, baseline_key):
        model_mae = results.get(model_key, {}).get("mae")
        base_mae = results.get(baseline_key, {}).get("mae")
        if model_mae is None or base_mae is None:
            return None
        return bool(model_mae < base_mae)

    manifest["model_readiness"] = {
        "overall_status": "research_stage_not_production_ready",
        "benchmark_warning": (
            "The persistence baseline outperforms every reported learned SOH "
            "model. Do not describe any model as production-ready."
        ),
        "uncertainty_quantification": "none",
        "calibration": "none",
        "models": {
            "nmc_lgbm": {
                "status": "experimental_research_stage",
                "test_n": results.get("nmc_lgbm_test", {}).get("n"),
                "test_mae": results.get("nmc_lgbm_test", {}).get("mae"),
                "persistence_mae": results.get("nmc_persistence_test", {}).get("mae"),
                "outperforms_persistence": _beats_persistence(
                    "nmc_lgbm_test", "nmc_persistence_test"
                ),
            },
            "lfp_lgbm": {
                "status": "experimental_research_stage_conditional_labels",
                "test_n": results.get("lfp_lgbm_test", {}).get("n"),
                "test_mae": results.get("lfp_lgbm_test", {}).get("mae"),
                "persistence_mae": results.get("lfp_persistence_test", {}).get("mae"),
                "outperforms_persistence": _beats_persistence(
                    "lfp_lgbm_test", "lfp_persistence_test"
                ),
            },
            "nmc_bilstm": {
                "status": "experimental_research_stage",
                "test_n": results.get("nmc_bilstm_test", {}).get("n"),
                "test_mae": results.get("nmc_bilstm_test", {}).get("mae"),
                "note": (
                    "Compare against persistence on the identical sequence test "
                    "subset; see INTEGRATION_CONTRACT.md."
                ),
            },
            "nmc_rul_lgbm": {
                "status": "unvalidated_pre_eol",
                "test_n": results.get("nmc_rul_test", {}).get("n"),
                "test_mae": results.get("nmc_rul_test", {}).get("mae"),
                "note": (
                    "All chronological test targets are post-EOL (RUL=0); no "
                    "pre-EOL forecasting capability is established."
                ),
            },
        },
    }
    
    # ── Sample Input/Output ──
    if len(nmc_test) > 0:
        sample_row = nmc_test.iloc[0]
        sample_input = {col: float(sample_row[col]) for col in nmc_feature_cols}
        sample_pred = float(lgbm_nmc.predict(
            np.array([[sample_row[c] for c in nmc_feature_cols]])
        )[0])
        sample_actual = float(sample_row["target_soh_pct"])
        
        manifest["sample_io"] = {
            "chemistry": "NMC",
            "model": "nmc_lgbm",
            "input_features": sample_input,
            "predicted_soh_pct": round(sample_pred, 4),
            "actual_soh_pct": round(sample_actual, 4),
            "horizon_cycles": 1,
            "target_semantics": (
                "SOH of the next observed cycle record (positional shift 1), "
                "not necessarily the next physical cycle."
            ),
            "model_status": "experimental_research_stage",
        }
    
    manifest_path = os.path.join(OUTPUT_DIR, "model_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved {manifest_path}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # [7/7] Print Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TRAINING & EVALUATION SUMMARY")
    print("=" * 70)
    
    for key, res in sorted(results.items()):
        print(f"  {res['label']}: MAE={res.get('mae', 'N/A')}, "
              f"RMSE={res.get('rmse', 'N/A')}, R²={res.get('r2', 'N/A')}, "
              f"n={res.get('n', 'N/A')}")
    
    print("\nArtifacts exported to:", OUTPUT_DIR)
    print("DONE.")
    
    return results, models, scalers


if __name__ == "__main__":
    main()
