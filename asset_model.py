import os
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Make torch optional for lightweight backends
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Optional metrics (used only during training/validation)
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None

# Dummy classes/functions if torch not available
from utils import logger
from risks import extract_basic_features_from_ohlcv
from fetchers import fetch_stock_history_yfinance

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
ASSET_LOGREG_PT = os.path.join(MODEL_DIR, "asset_logreg.pt")
ASSET_LOGREG_META = os.path.join(MODEL_DIR, "asset_logreg.meta.json")
ASSET_DUAL_PT = os.path.join(MODEL_DIR, "asset_dual.pt")
ASSET_DUAL_META = os.path.join(MODEL_DIR, "asset_dual.meta.json")


if TORCH_AVAILABLE:
    class LogisticRiskModel(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x: Any) -> Any:
            # returns logits
            return self.linear(x).squeeze(-1)
else:
    class LogisticRiskModel:
        def __init__(self, input_dim: int):
            self.input_dim = input_dim

        def forward(self, x):
            # Dummy: return zeros
            return np.zeros(x.shape[0])


def _window_features(df: pd.DataFrame, window: int) -> Dict[str, float]:
    """Compute a set of features on the last `window` rows of df using existing extractor
    and add a few extra predictors (returns, MA ratios)."""
    wdf = df.iloc[-window:] if len(df) >= window else df
    base = extract_basic_features_from_ohlcv(wdf)
    out = dict(base)
    # Robustly determine the close/price series (supports CoinGecko 'price')
    try:
        cols = {c.lower(): c for c in wdf.columns}
        close_col = cols.get("close") or cols.get("adj close") or cols.get("price")
        if close_col is None:
            try:
                numeric_cols = [c for c in wdf.columns if pd.api.types.is_numeric_dtype(wdf[c])]
                close_col = numeric_cols[-1] if numeric_cols else wdf.columns[-1]
            except Exception:
                close_col = wdf.columns[-1]
        close = wdf[close_col]
    except Exception:
        close = wdf.iloc[:, -1]
    # Returns
    try:
        r1 = float(close.pct_change().iloc[-1])
        if not np.isfinite(r1):
            r1 = 0.0
        out["ret_1"] = r1
    except Exception:
        out["ret_1"] = 0.0
    try:
        r5 = float(close.pct_change(5).iloc[-1])
        if not np.isfinite(r5):
            r5 = 0.0
        out["ret_5"] = r5
    except Exception:
        out["ret_5"] = 0.0
    # Moving average ratio
    try:
        ma10 = close.rolling(10, min_periods=1).mean().iloc[-1]
        ma20 = close.rolling(20, min_periods=1).mean().iloc[-1]
        out["ma_ratio_10_20"] = float(ma10 / (ma20 + 1e-12) - 1.0)
    except Exception:
        out["ma_ratio_10_20"] = 0.0
    # Sanitize None/non-finite values to keep feature matrix numeric
    for k, v in list(out.items()):
        try:
            vv = float(v)
            if not np.isfinite(vv):
                vv = 0.0
            out[k] = vv
        except Exception:
            out[k] = 0.0
    return out


def _get_close_series(df: pd.DataFrame) -> pd.Series:
    cols = {c.lower(): c for c in df.columns}
    close_col = cols.get("close") or cols.get("adj close") or cols.get("price")
    if close_col is None:
        try:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            close_col = numeric_cols[-1] if numeric_cols else df.columns[-1]
        except Exception:
            close_col = df.columns[-1]
    return df[close_col]


def _future_drawdown_label(df: pd.DataFrame, horizon: int, thresh: float) -> int:
    """Label 1 if next `horizon` bars experience drawdown <= `thresh` relative to last close."""
    if len(df) < horizon + 1:
        return 0
    close = _get_close_series(df)
    last_close = float(close.iloc[-horizon-1])  # at time t
    future = close.iloc[-horizon:]
    dd = float(future.min() / (last_close + 1e-12) - 1.0)
    return 1 if dd <= thresh else 0


def build_asset_dataset(tickers: List[str], period: str = "6mo", window: int = 30, horizon: int = 5,
                        drawdown_thresh: float = -0.05) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build a dataset of windowed features and future drawdown labels from yfinance data."""
    X_list: List[List[float]] = []
    y_list: List[int] = []
    feature_names: List[str] = []

    for symbol in tickers:
        try:
            df = fetch_stock_history_yfinance(symbol, period=period)
            if df is None or len(df) < window + horizon + 5:
                logger.warning("Skipping %s: insufficient data", symbol)
                continue
            # Slide a window through the series
            for end in range(window, len(df) - horizon):
                sub = df.iloc[: end + horizon]  # include future for labeling
                wsub = sub.iloc[end - window: end]  # window slice for features
                feats = _window_features(wsub, window)
                if not feature_names:
                    feature_names = list(feats.keys())
                X_list.append([feats[k] for k in feature_names])
                label = _future_drawdown_label(sub.iloc[end - 1: end + horizon], horizon=horizon, thresh=drawdown_thresh)
                y_list.append(label)
        except Exception:
            logger.exception("Failed to process %s", symbol)
            continue

    if not X_list:
        raise RuntimeError("No training samples collected. Check tickers/period/network.")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y, feature_names


def train_asset_logreg(tickers: List[str] | None = None, period: str = "6mo", window: int = 30, horizon: int = 5,
                       drawdown_thresh: float = -0.05, epochs: int = 10, lr: float = 1e-2) -> str:
    """Train a simple logistic regression risk model and save to models directory.

    Returns path to the saved .pt file. A meta JSON with feature_names and normalization is also saved.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")
    if tickers is None:
        tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]

    X, y, feature_names = build_asset_dataset(tickers, period=period, window=window, horizon=horizon,
                                              drawdown_thresh=drawdown_thresh)
    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xn = (X - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogisticRiskModel(input_dim=Xn.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # pos_weight to handle imbalance
    pos = float(y.sum() + 1e-6)
    neg = float((len(y) - y.sum()) + 1e-6)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tensor = torch.tensor(Xn, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(X_tensor)
        loss = loss_fn(logits, y_tensor)
        loss.backward()
        opt.step()
        if (epoch + 1) % 2 == 0:
            logger.info("Epoch %d/%d - loss: %.4f", epoch + 1, epochs, float(loss.item()))

    # Save model and meta
    torch.save(model.state_dict(), ASSET_LOGREG_PT)
    meta = {
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "window": window,
        "horizon": horizon,
        "drawdown_thresh": drawdown_thresh,
    }
    with open(ASSET_LOGREG_META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)

    logger.info("Saved asset logreg model to %s and meta to %s", ASSET_LOGREG_PT, ASSET_LOGREG_META)
    return ASSET_LOGREG_PT


def load_asset_logreg(device: Any | None = None) -> tuple[LogisticRiskModel, Dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise FileNotFoundError("Asset model artifacts not found")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ASSET_LOGREG_PT) or not os.path.exists(ASSET_LOGREG_META):
        raise FileNotFoundError("Asset model artifacts not found")
    with open(ASSET_LOGREG_META, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    model = LogisticRiskModel(input_dim=len(meta["feature_names"]))
    model.load_state_dict(torch.load(ASSET_LOGREG_PT, map_location=device))
    model.to(device)
    model.eval()
    return model, meta


def infer_asset_logreg_from_df(df: pd.DataFrame, model: LogisticRiskModel, meta: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
    """Return (probability_of_future_drawdown, contribs_by_feature)"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")
    feats = _window_features(df, meta.get("window", 30))
    # Ensure vector ordering
    xs = np.array([feats.get(k, 0.0) for k in meta["feature_names"]], dtype=np.float32)
    # Sanitize potential NaNs/Infs from short windows
    xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.array(meta["mean"], dtype=np.float32)
    std = np.array(meta["std"], dtype=np.float32)
    xsn = (xs - mean) / std
    xsn = np.nan_to_num(xsn, nan=0.0, posinf=0.0, neginf=0.0)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(xsn, dtype=torch.float32, device=device))
        prob = torch.sigmoid(logits).item()
    # Simple contribution estimate: weight * standardized value
    w = model.linear.weight.detach().cpu().numpy().reshape(-1)
    contribs = {k: float(w[i] * xsn[i]) for i, k in enumerate(meta["feature_names"])}
    return float(prob), contribs


# ----------------------------
# Dual-head model and training
# ----------------------------

if TORCH_AVAILABLE:
    class DualLogisticSignalModel(nn.Module):
        """Two-output linear model:
        - output[0]: drawdown event (1 if drawdown <= drawdown_thresh)
        - output[1]: up move event (1 if horizon return >= up_return_thresh)
        """

        def __init__(self, input_dim: int):
            super().__init__()
            self.linear = nn.Linear(input_dim, 2)

        def forward(self, x: Any) -> Any:
            return self.linear(x)
else:
    class DualLogisticSignalModel:
        def __init__(self, input_dim: int):
            self.input_dim = input_dim


def _future_up_label(df: pd.DataFrame, horizon: int, ret_thresh: float) -> int:
    """Label 1 if close at t+horizon is >= (1+ret_thresh)*close at t."""
    if len(df) < horizon + 1:
        return 0
    close = _get_close_series(df)
    last_close = float(close.iloc[-horizon - 1])  # at time t
    future_last = float(close.iloc[-1])  # at time t+horizon
    ret = future_last / (last_close + 1e-12) - 1.0
    return 1 if ret >= ret_thresh else 0


def build_asset_dataset_dual(
    tickers: List[str],
    period: str = "6mo",
    window: int = 30,
    horizon: int = 5,
    drawdown_thresh: float = -0.05,
    up_return_thresh: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build dataset returning features and two labels: (y_down, y_up)."""
    X_list: List[List[float]] = []
    y_down_list: List[int] = []
    y_up_list: List[int] = []
    feature_names: List[str] = []

    for symbol in tickers:
        try:
            df = fetch_stock_history_yfinance(symbol, period=period)
            if df is None or len(df) < window + horizon + 5:
                logger.warning("Skipping %s: insufficient data", symbol)
                continue
            for end in range(window, len(df) - horizon):
                sub = df.iloc[: end + horizon]
                wsub = sub.iloc[end - window : end]
                feats = _window_features(wsub, window)
                if not feature_names:
                    feature_names = list(feats.keys())
                X_list.append([feats[k] for k in feature_names])
                slice_for_label = sub.iloc[end - 1 : end + horizon]
                y_down_list.append(_future_drawdown_label(slice_for_label, horizon=horizon, thresh=drawdown_thresh))
                y_up_list.append(_future_up_label(slice_for_label, horizon=horizon, ret_thresh=up_return_thresh))
        except Exception:
            logger.exception("Failed to process %s", symbol)
            continue

    if not X_list:
        raise RuntimeError("No training samples collected. Check tickers/period/network.")

    X = np.asarray(X_list, dtype=np.float32)
    y_down = np.asarray(y_down_list, dtype=np.float32)
    y_up = np.asarray(y_up_list, dtype=np.float32)
    return X, y_down, y_up, feature_names


def _tune_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pick a probability threshold that maximizes F1 over a simple grid."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (scores >= t).astype(np.int32)
        tp = float(((pred == 1) & (labels == 1)).sum())
        fp = float(((pred == 1) & (labels == 0)).sum())
        fn = float(((pred == 0) & (labels == 1)).sum())
        if tp + fp + fn == 0:
            f1 = 0.0
        else:
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train_asset_dual(
    tickers: List[str] | None = None,
    period: str = "6mo",
    window: int = 30,
    horizon: int = 5,
    drawdown_thresh: float = -0.05,
    up_return_thresh: float = 0.0,
    epochs: int = 10,
    lr: float = 1e-2,
    val_split: float = 0.2,
    seed: int = 42,
) -> str:
    """Train a dual-head logistic model for risk (down) and direction (up)."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")
    if tickers is None:
        tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BTC-USD", "ETH-USD", "SOL-USD"]

    X, y_down, y_up, feature_names = build_asset_dataset_dual(
        tickers, period=period, window=window, horizon=horizon,
        drawdown_thresh=drawdown_thresh, up_return_thresh=up_return_thresh
    )
    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xn = (X - mean) / std

    # Train/val split
    rs = np.random.RandomState(seed)
    idx = np.arange(len(Xn))
    rs.shuffle(idx)
    split = int(len(idx) * (1 - val_split))
    tr_idx, va_idx = idx[:split], idx[split:]
    Xtr, Xva = Xn[tr_idx], Xn[va_idx]
    ytr_down, ytr_up = y_down[tr_idx], y_up[tr_idx]
    yva_down, yva_up = y_down[va_idx], y_up[va_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualLogisticSignalModel(input_dim=Xn.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # class imbalance handling
    pos_down = float(ytr_down.sum() + 1e-6)
    neg_down = float((len(ytr_down) - ytr_down.sum()) + 1e-6)
    pos_up = float(ytr_up.sum() + 1e-6)
    neg_up = float((len(ytr_up) - ytr_up.sum()) + 1e-6)
    pos_weight = torch.tensor([neg_down / pos_down, neg_up / pos_up], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(np.stack([ytr_down, ytr_up], axis=1), dtype=torch.float32, device=device)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = loss_fn(logits, ytr_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 2 == 0:
            logger.info("[dual] Epoch %d/%d - loss: %.4f", epoch + 1, epochs, float(loss.item()))

    # Validation metrics
    model.eval()
    with torch.no_grad():
        Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
        logits_va = model(Xva_t).cpu().numpy()
        probs_va = 1.0 / (1.0 + np.exp(-logits_va))
        s_down, s_up = probs_va[:, 0], probs_va[:, 1]
        try:
            auc_down = float(roc_auc_score(yva_down, s_down))
        except Exception:
            auc_down = float("nan")
        try:
            auc_up = float(roc_auc_score(yva_up, s_up))
        except Exception:
            auc_up = float("nan")
        try:
            ap_down = float(average_precision_score(yva_down, s_down))
        except Exception:
            ap_down = float("nan")
        try:
            ap_up = float(average_precision_score(yva_up, s_up))
        except Exception:
            ap_up = float("nan")
        thr_down = _tune_threshold(s_down, yva_down)
        thr_up = _tune_threshold(s_up, yva_up)

    # Save artifacts
    torch.save(model.state_dict(), ASSET_DUAL_PT)
    meta = {
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "window": window,
        "horizon": horizon,
        "drawdown_thresh": drawdown_thresh,
        "up_return_thresh": up_return_thresh,
        "class_thresholds": {"down": float(thr_down), "up": float(thr_up)},
        "metrics": {
            "val_auc_down": auc_down,
            "val_auc_up": auc_up,
            "val_ap_down": ap_down,
            "val_ap_up": ap_up,
        },
    }
    with open(ASSET_DUAL_META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    logger.info(
        "Saved asset dual model to %s and meta to %s (AUC down=%.3f, up=%.3f)",
        ASSET_DUAL_PT,
        ASSET_DUAL_META,
        meta["metrics"]["val_auc_down"],
        meta["metrics"]["val_auc_up"],
    )
    return ASSET_DUAL_PT


def load_asset_dual(device: Any | None = None) -> tuple[DualLogisticSignalModel, Dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise FileNotFoundError("Asset dual model artifacts not found")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ASSET_DUAL_PT) or not os.path.exists(ASSET_DUAL_META):
        raise FileNotFoundError("Asset dual model artifacts not found")
    with open(ASSET_DUAL_META, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    model = DualLogisticSignalModel(input_dim=len(meta["feature_names"]))
    model.load_state_dict(torch.load(ASSET_DUAL_PT, map_location=device))
    model.to(device)
    model.eval()
    return model, meta


def infer_asset_dual_from_df(
    df: pd.DataFrame, model: DualLogisticSignalModel, meta: Dict[str, Any]
) -> tuple[float, float, Dict[str, float], Dict[str, float]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available")
    feats = _window_features(df, meta.get("window", 30))
    xs = np.array([feats.get(k, 0.0) for k in meta["feature_names"]], dtype=np.float32)
    xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.array(meta["mean"], dtype=np.float32)
    std = np.array(meta["std"], dtype=np.float32)
    xsn = (xs - mean) / std
    xsn = np.nan_to_num(xsn, nan=0.0, posinf=0.0, neginf=0.0)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(xsn, dtype=torch.float32, device=device))
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    w = model.linear.weight.detach().cpu().numpy()  # shape (2, D)
    contrib_down = {k: float(w[0, i] * xsn[i]) for i, k in enumerate(meta["feature_names"])}
    contrib_up = {k: float(w[1, i] * xsn[i]) for i, k in enumerate(meta["feature_names"])}
    return float(probs[0]), float(probs[1]), contrib_down, contrib_up
