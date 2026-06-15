"""
ATAD 快速特征提取模块 - 优化版本
Fast feature extraction for ATAD with vectorized operations.
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import stats
from scipy.fft import fft, fftfreq
import warnings

warnings.filterwarnings('ignore')


def detect_period_fft(values: np.ndarray) -> int:
    """使用FFT检测主导周期"""
    n = len(values)
    if n < 16:
        return max(n // 2, 1)
    detrended = values - np.mean(values)
    fft_vals = fft(detrended)
    freqs = fftfreq(n)
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    mag = np.abs(fft_vals[pos_mask])
    if len(mag) > 1:
        dom = np.argmax(mag[1:]) + 1
        if freqs[dom] > 0:
            p = int(1.0 / freqs[dom])
            return max(2, min(p, n // 3))
    return max(n // 10, 2)


# =============================================================================
# Fast rolling statistics using cumulative sums
# =============================================================================

def rolling_mean_fast(values: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling mean."""
    n = len(values)
    result = np.zeros(n)
    if n == 0 or window <= 0:
        return result

    cumsum = np.cumsum(np.insert(values, 0, 0))
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        result[i] = (cumsum[i + 1] - cumsum[start]) / count
    return result


def rolling_var_fast(values: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling variance."""
    n = len(values)
    result = np.zeros(n)
    if n == 0 or window <= 0:
        return result

    cumsum = np.cumsum(np.insert(values, 0, 0))
    cumsum2 = np.cumsum(np.insert(values ** 2, 0, 0))

    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        if count < 2:
            result[i] = 0
        else:
            mean = (cumsum[i + 1] - cumsum[start]) / count
            mean_sq = (cumsum2[i + 1] - cumsum2[start]) / count
            var = mean_sq - mean ** 2
            result[i] = max(var, 0)
    return result


def rolling_acf1_fast(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling first-order autocorrelation."""
    n = len(values)
    result = np.zeros(n)
    for i in range(1, n):
        start = max(0, i - window + 1)
        win = values[start:i + 1]
        if len(win) < 2:
            continue
        demean = win - np.mean(win)
        denom = np.sum(demean ** 2)
        if denom < 1e-10:
            result[i] = 0
        else:
            result[i] = np.sum(demean[:-1] * demean[1:]) / denom
    return result


def rolling_trend_strength(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling strength of trend (simplified)."""
    n = len(values)
    result = np.zeros(n)
    for i in range(window, n):
        win = values[i - window:i + 1]
        t = np.arange(len(win))
        if np.std(win) < 1e-10:
            result[i] = 0
        else:
            slope, _, r_val, _, _ = stats.linregress(t, win)
            result[i] = r_val ** 2
    return result


def rolling_crossingpoint(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling zero-crossing count."""
    n = len(values)
    result = np.zeros(n)
    for i in range(2, n):
        start = max(0, i - window + 1)
        win = values[start:i + 1]
        if len(win) < 2:
            continue
        centered = win - np.mean(win)
        result[i] = np.sum((centered[:-1] * centered[1:]) < 0)
    return result


def rolling_entropy(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling spectral entropy."""
    n = len(values)
    result = np.zeros(n)
    for i in range(window, n):
        win = values[i - window:i + 1]
        if len(win) < 4:
            continue
        fft_vals = np.abs(fft(win - np.mean(win)))
        fft_vals = fft_vals[:len(fft_vals) // 2]
        total = np.sum(fft_vals)
        if total < 1e-10:
            continue
        psd = fft_vals / total
        psd = psd[psd > 0]
        result[i] = -np.sum(psd * np.log(psd))
    return result


# =============================================================================
# EWMA-based fluctuation prediction (shared with FluxEV approach)
# =============================================================================

def ewma_prediction(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Vectorized EWMA prediction for the entire series."""
    n = len(values)
    pred = np.zeros(n)
    if n == 0:
        return pred

    # EWMA smoothed values
    ewma = np.zeros(n)
    ewma[0] = values[0]
    for i in range(1, n):
        ewma[i] = alpha * values[i] + (1 - alpha) * ewma[i - 1]

    # Prediction = EWMA_{i-1} (prediction for i based on past)
    pred[1:] = ewma[:-1]
    return pred


# =============================================================================
# Fast Feature Extraction
# =============================================================================

def extract_features_fast(
    values: np.ndarray,
    labels: Optional[np.ndarray] = None,
    period: Optional[int] = None,
    wpr: int = 24,
    verbose: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Fast feature extraction for ATAD using vectorized operations.

    Extracts 37 features per data point.
    Features are computed efficiently with pre-computed rolling statistics.
    """
    n = len(values)

    if period is None:
        period = detect_period_fft(values)

    # Window sizes
    p = max(period, 4)
    ws = max(p, 10)
    w1 = max(p // 4, 3)
    w2 = max(p // 2, 6)
    w3 = max(p, 12)

    if verbose:
        print(f"  Extracting features for {n} points (period={p}, ws={ws})...")

    feature_names = [
        'original_value',
        # Statistical (7 features)
        'mean', 'var', 'crossingpoint', 'acf1', 'trend',
        'entropy', 'value_diff',
        # Forecasting error based features (5 metrics × 3 windows = 15)
        'ME_w1', 'RMSE_w1', 'MAE_w1', 'MPE_w1', 'MAPE_w1',
        'ME_w2', 'RMSE_w2', 'MAE_w2', 'MPE_w2', 'MAPE_w2',
        'ME_w3', 'RMSE_w3', 'MAE_w3', 'MPE_w3', 'MAPE_w3',
        # Temporal features (14)
        'max_level_shift', 'max_var_shift',
        'diff_half_p', 'diff_p', 'diff_2p', 'diff_wpr', 'diff_half_wpr',
        'rolling_mean', 'rolling_std', 'rolling_max', 'rolling_min',
        'rolling_skew', 'rolling_kurt', 'rolling_iqr', 'ewma_residual',
    ]

    features = np.zeros((n, len(feature_names)))

    # ---- Pre-compute rolling statistics ----
    if verbose:
        print("    Computing rolling statistics...")

    roll_mean = rolling_mean_fast(values, ws)
    roll_var = rolling_var_fast(values, ws)
    roll_acf1 = rolling_acf1_fast(values, ws)
    roll_trend = rolling_trend_strength(values, ws)
    roll_cross = rolling_crossingpoint(values, ws)
    roll_ent = rolling_entropy(values, ws)

    # EWMA prediction and residuals
    ewma_pred = ewma_prediction(values, alpha=0.3)
    ewma_residual = values - ewma_pred

    if verbose:
        print("    Computing forecasting errors...")

    # ---- Forecasting error features ----
    # Pre-compute for efficiency using rolling windows of EWMA residuals
    for w_idx, w_size in enumerate([w1, w2, w3]):
        prefix = ['w1', 'w2', 'w3'][w_idx]
        base_col = 8 + w_idx * 5  # column offset

        for i in range(n):
            start = max(0, i - w_size + 1)
            win = values[start:i + 1]
            pred_win = ewma_pred[start:i + 1]
            errs = win - pred_win
            if len(errs) < 2:
                continue

            features[i, base_col] = np.mean(errs)            # ME
            features[i, base_col + 1] = np.sqrt(np.mean(errs ** 2))  # RMSE
            features[i, base_col + 2] = np.mean(np.abs(errs))  # MAE
            # MPE / MAPE
            mask = np.abs(win) > 1e-10
            if np.any(mask):
                features[i, base_col + 3] = np.mean(errs[mask] / np.abs(win[mask])) * 100  # MPE
                features[i, base_col + 4] = np.mean(np.abs(errs[mask] / np.abs(win[mask]))) * 100  # MAPE

    if verbose:
        print("    Computing temporal features...")

    # ---- Fast rolling stats for kurtosis/skew ----
    roll_std_arr = np.sqrt(roll_var + 1e-10)
    roll_max_arr = np.zeros(n)
    roll_min_arr = np.zeros(n)
    roll_iqr_arr = np.zeros(n)
    roll_skew_arr = np.zeros(n)
    roll_kurt_arr = np.zeros(n)

    for i in range(n):
        start = max(0, i - ws + 1)
        win = values[start:i + 1]
        if len(win) >= 4:
            roll_max_arr[i] = np.max(win)
            roll_min_arr[i] = np.min(win)
            roll_iqr_arr[i] = np.percentile(win, 75) - np.percentile(win, 25)
            # Skewness
            m = np.mean(win)
            s = np.std(win, ddof=0)
            if s > 1e-10:
                roll_skew_arr[i] = np.mean(((win - m) / s) ** 3)
                roll_kurt_arr[i] = np.mean(((win - m) / s) ** 4) - 3

    # ---- Max level/var shift ----
    half_ws = ws // 2
    level_shift = np.zeros(n)
    var_shift = np.zeros(n)
    for i in range(half_ws * 2, n):
        win1 = values[i - half_ws * 2:i - half_ws]
        win2 = values[i - half_ws:i]
        if len(win1) >= 3 and len(win2) >= 3:
            t1 = np.sort(win1)[1:-1]
            t2 = np.sort(win2)[1:-1]
            level_shift[i] = abs(np.mean(t1) - np.mean(t2))
            var_shift[i] = abs(np.var(win1, ddof=1) - np.var(win2, ddof=1))

    # ---- Diff features ----
    half_p = max(p // 2, 1)
    diff_half_p_arr = np.zeros(n)
    diff_p_arr = np.zeros(n)
    diff_2p_arr = np.zeros(n)
    diff_wpr_arr = np.zeros(n)
    diff_half_wpr_arr = np.zeros(n)

    for i in range(n):
        if i >= half_p:
            diff_half_p_arr[i] = values[i] - values[i - half_p]
        if i >= p:
            diff_p_arr[i] = values[i] - values[i - p]
        if i >= 2 * p:
            diff_2p_arr[i] = values[i] - values[i - 2 * p]
        if i >= wpr:
            diff_wpr_arr[i] = values[i] - values[i - wpr]
        if i >= wpr // 2:
            diff_half_wpr_arr[i] = values[i] - values[i - wpr // 2]

    # ---- Assemble features ----
    if verbose:
        print("    Assembling feature matrix...")

    for i in range(n):
        col = 0
        features[i, col] = values[i]; col += 1

        # Statistical
        features[i, col] = roll_mean[i]; col += 1
        features[i, col] = roll_var[i]; col += 1
        features[i, col] = roll_cross[i]; col += 1
        features[i, col] = roll_acf1[i]; col += 1
        features[i, col] = roll_trend[i]; col += 1
        features[i, col] = roll_ent[i]; col += 1
        features[i, col] = values[i] - values[i - 1] if i > 0 else 0; col += 1
        # col is now 8, forecasting errors already set above
        col = 23  # skip past forecasting error slots (8 + 15 = 23)

        # Temporal
        features[i, col] = level_shift[i]; col += 1
        features[i, col] = var_shift[i]; col += 1
        features[i, col] = diff_half_p_arr[i]; col += 1
        features[i, col] = diff_p_arr[i]; col += 1
        features[i, col] = diff_2p_arr[i]; col += 1
        features[i, col] = diff_wpr_arr[i]; col += 1
        features[i, col] = diff_half_wpr_arr[i]; col += 1

        # Additional stats
        features[i, col] = roll_mean[i]; col += 1
        features[i, col] = roll_std_arr[i]; col += 1
        features[i, col] = roll_max_arr[i]; col += 1
        features[i, col] = roll_min_arr[i]; col += 1
        features[i, col] = roll_skew_arr[i]; col += 1
        features[i, col] = roll_kurt_arr[i]; col += 1
        features[i, col] = roll_iqr_arr[i]; col += 1
        features[i, col] = ewma_residual[i] if i < len(ewma_residual) else 0; col += 1

    # Handle NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        print(f"    Feature extraction complete: {features.shape[1]} features × {features.shape[0]} points")

    return features, labels, feature_names


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"F:\1")
    from data_loader import load_training_data
    import time

    print("Testing fast feature extraction...")
    train = load_training_data(r"F:\1\jmeter_avg_elapsed_ms(1).csv")
    values = train['value'].values[:500]

    start = time.time()
    feats, labels, names = extract_features_fast(values, train['label'].values[:500])
    elapsed = time.time() - start

    print(f"Features: {feats.shape}")
    print(f"Time for 500 points: {elapsed:.2f}s")
    print(f"Estimated time for 4300 points: {elapsed * 4300/500:.1f}s")
    print(f"Feature names: {names}")
