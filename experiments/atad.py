"""
ATAD (Active Transfer Anomaly Detection) 算法实现
论文: Cross-dataset Time Series Anomaly Detection for Cloud Systems (USENIX ATC'19)

核心组件:
1. 特征提取 (37个特征): 统计特征、预测误差特征、时序特征
2. 迁移学习: K-means聚类 + CORAL特征对齐 + 随机森林分类器
3. 主动学习: UCD (Uncertainty-Context Diversity) 采样策略
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from collections import deque
import warnings

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.fft import fft, fftfreq

# Fast feature extraction (separate module for performance)
try:
    from atad_features import extract_features_fast, detect_period_fft
except ImportError:
    # Fallback: define inline if module not found
    def extract_features_fast(*args, **kwargs):
        raise RuntimeError("atad_features module not found")
    def detect_period_fft(*args, **kwargs):
        raise RuntimeError("atad_features module not found")

warnings.filterwarnings('ignore')


# =============================================================================
# Feature Extraction (37 features)
# =============================================================================

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


def _rolling_window(values: np.ndarray, idx: int, window_size: int, lookback: bool = True):
    """Get values in a rolling window around idx."""
    if lookback:
        start = max(0, idx - window_size + 1)
        end = idx + 1
    else:
        start = idx
        end = min(len(values), idx + window_size)
    return values[start:end]


# ---------- Statistical Features (11) ----------

def feat_mean(values: np.ndarray, idx: int, window_size: int) -> float:
    win = _rolling_window(values, idx, window_size)
    return np.mean(win) if len(win) > 0 else 0


def feat_var(values: np.ndarray, idx: int, window_size: int) -> float:
    win = _rolling_window(values, idx, window_size)
    return np.var(win) if len(win) > 1 else 0


def feat_crossingpoint(values: np.ndarray, idx: int, window_size: int) -> int:
    """Number of zero-crossings after mean centering."""
    win = _rolling_window(values, idx, window_size)
    if len(win) < 2:
        return 0
    centered = win - np.mean(win)
    count = np.sum((centered[:-1] * centered[1:]) < 0)
    return count


def feat_acf1(values: np.ndarray, idx: int, window_size: int) -> float:
    """First order autocorrelation."""
    win = _rolling_window(values, idx, window_size)
    if len(win) < 2:
        return 0
    demean = win - np.mean(win)
    denom = np.sum(demean ** 2)
    if denom < 1e-10:
        return 0
    return np.sum(demean[:-1] * demean[1:]) / denom


def _stl_decompose(series: np.ndarray, period: int) -> Dict:
    """Simplified STL decomposition."""
    n = len(series)
    if n < period * 2:
        return {'trend': np.zeros(n), 'seasonal': np.zeros(n), 'remainder': series - np.mean(series)}

    # Trend: moving average
    trend = np.convolve(series, np.ones(period) / period, mode='same')
    # Seasonal: average of detrended values at same position
    detrended = series - trend
    seasonal = np.zeros(n)
    if n >= period:
        for i in range(period):
            indices = list(range(i, n, period))
            if indices:
                avg = np.mean(detrended[indices])
                for j in indices:
                    seasonal[j] = avg
    remainder = series - trend - seasonal
    return {'trend': trend, 'seasonal': seasonal, 'remainder': remainder}


def feat_acfremainder(values: np.ndarray, idx: int, window_size: int, period: int) -> float:
    """ACF of STL remainder."""
    win = _rolling_window(values, idx, window_size)
    if len(win) < 2:
        return 0
    decomp = _stl_decompose(win, period)
    rem = decomp['remainder']
    return feat_acf1(rem, len(rem) - 1, len(rem))


def feat_trend(values: np.ndarray, idx: int, window_size: int, period: int) -> float:
    """Strength of trend."""
    win = _rolling_window(values, idx, window_size)
    if len(win) < 2:
        return 0
    decomp = _stl_decompose(win, period)
    var_rem = np.var(decomp['remainder'])
    var_trend = np.var(decomp['trend'])
    denom = var_rem + var_trend
    return max(0, 1 - var_rem / denom) if denom > 1e-10 else 0


def feat_linearity(values: np.ndarray, idx: int, window_size: int, period: int) -> float:
    """Linearity computed on trend of STL."""
    win = _rolling_window(values, idx, window_size)
    if len(win) < 2:
        return 0
    decomp = _stl_decompose(win, period)
    trend = decomp['trend']
    t = np.arange(len(trend))
    if np.std(trend) < 1e-10:
        return 0
    slope, _, r_val, _, _ = stats.linregress(t, trend)
    return r_val ** 2


def feat_curvature(values: np.ndarray, idx: int, window_size: int, period: int) -> float:
    """Curvature computed on trend of STL."""
    win = _rolling_window(values, idx, window_size)
    if len(win) < 3:
        return 0
    decomp = _stl_decompose(win, period)
    trend = decomp['trend']
    t = np.arange(len(trend))
    try:
        coeffs = np.polyfit(t, trend, 2)
        return abs(coeffs[0])
    except:
        return 0


def feat_entropy(values: np.ndarray, idx: int, window_size: int) -> float:
    """Spectral entropy."""
    win = _rolling_window(values, idx, window_size)
    if len(win) < 4:
        return 0
    fft_vals = np.abs(fft(win - np.mean(win)))
    fft_vals = fft_vals[:len(fft_vals) // 2]
    total = np.sum(fft_vals)
    if total < 1e-10:
        return 0
    psd = fft_vals / total
    psd = psd[psd > 0]
    return -np.sum(psd * np.log(psd))


def feat_arch_test(values: np.ndarray, idx: int, window_size: int) -> float:
    """P-value of LM test for ARCH effect."""
    win = _rolling_window(values, idx, window_size)
    if len(win) < 10:
        return 1.0
    try:
        squared = (win - np.mean(win)) ** 2
        n = len(squared)
        # Simple ARCH LM test
        X = np.column_stack([np.ones(n - 1), squared[:-1]])
        y = squared[1:]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        r2 = 1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)
        lm_stat = n * r2
        return 1 - stats.chi2.cdf(lm_stat, 1)
    except:
        return 1.0


def feat_garch_test(values: np.ndarray, idx: int, window_size: int) -> float:
    """P-value of LM test for GARCH effect."""
    # Simplified; for GARCH(1,1), test is similar with more lags
    win = _rolling_window(values, idx, window_size)
    if len(win) < 10:
        return 1.0
    try:
        squared = (win - np.mean(win)) ** 2
        n = len(squared)
        X = np.column_stack([np.ones(n - 2), squared[1:-1], squared[:-2]])
        y = squared[2:]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        r2 = 1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)
        lm_stat = n * r2
        return 1 - stats.chi2.cdf(lm_stat, 2)
    except:
        return 1.0


# ---------- Forecasting Error Features (5 metrics × 3 windows) ----------

def _simple_holt_predict(values: np.ndarray, alpha: float = 0.3, beta: float = 0.1) -> float:
    """Simple Holt's linear trend prediction."""
    if len(values) < 2:
        return values[-1] if len(values) > 0 else 0
    level = values[0]
    trend = values[1] - values[0]
    for i in range(1, len(values)):
        prev_level = level
        level = alpha * values[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return level + trend


def _simple_stl_predict(values: np.ndarray, period: int) -> float:
    """Simplified prediction using STL decomposition."""
    if len(values) <= period:
        return np.mean(values)
    decomp = _stl_decompose(values, period)
    # Last trend + same seasonal position
    last_trend = decomp['trend'][-1]
    seasonal_pos = len(values) % period
    if seasonal_pos < len(decomp['seasonal']):
        return last_trend + decomp['seasonal'][seasonal_pos]
    return last_trend


def _ensemble_predict(values: np.ndarray, period: int) -> float:
    """Ensemble prediction combining Holt and STL."""
    pred_holt = _simple_holt_predict(values)
    pred_stl = _simple_stl_predict(values[:-1], period) if len(values) > period else pred_holt
    # Equal weight ensemble
    return (pred_holt + pred_stl) / 2.0


def _calc_forecast_error_features(values: np.ndarray, idx: int, window_size: int, period: int) -> Dict:
    """Calculate forecasting error features."""
    win = _rolling_window(values, idx, window_size)
    if len(win) < 3:
        return {'ME': 0, 'RMSE': 0, 'MAE': 0, 'MPE': 0, 'MAPE': 0}

    errors = []
    for i in range(3, len(win)):
        pred = _ensemble_predict(win[:i], period)
        actual = win[i]
        errors.append(actual - pred)

    if not errors:
        return {'ME': 0, 'RMSE': 0, 'MAE': 0, 'MPE': 0, 'MAPE': 0}

    err_arr = np.array(errors)
    actual_arr = win[3:]

    me = np.mean(err_arr)
    rmse = np.sqrt(np.mean(err_arr ** 2))
    mae = np.mean(np.abs(err_arr))
    mpe = np.mean(err_arr / np.where(np.abs(actual_arr) > 1e-10, actual_arr, 1.0)) * 100
    mape = np.mean(np.abs(err_arr / np.where(np.abs(actual_arr) > 1e-10, actual_arr, 1.0))) * 100

    return {'ME': me, 'RMSE': rmse, 'MAE': mae, 'MPE': mpe, 'MAPE': mape}


# ---------- Temporal Features ----------

def feat_max_level_shift(values: np.ndarray, idx: int, window_size: int) -> float:
    """Max trimmed mean between two consecutive windows."""
    half = window_size // 2
    if idx < half * 2:
        return 0
    win1 = values[idx - half * 2:idx - half]
    win2 = values[idx - half:idx]
    if len(win1) < 3 or len(win2) < 3:
        return 0
    trim1 = np.sort(win1)[1:-1] if len(win1) > 2 else win1
    trim2 = np.sort(win2)[1:-1] if len(win2) > 2 else win2
    return abs(np.mean(trim1) - np.mean(trim2))


def feat_max_var_shift(values: np.ndarray, idx: int, window_size: int) -> float:
    """Max variance shift between two consecutive windows."""
    half = window_size // 2
    if idx < half * 2:
        return 0
    win1 = values[idx - half * 2:idx - half]
    win2 = values[idx - half:idx]
    v1 = np.var(win1) if len(win1) > 1 else 0
    v2 = np.var(win2) if len(win2) > 1 else 0
    return abs(v1 - v2)


def feat_max_kl_shift(values: np.ndarray, idx: int, window_size: int, n_bins: int = 10) -> float:
    """Max KL divergence shift between two consecutive windows."""
    half = window_size // 2
    if idx < half * 2:
        return 0
    win1 = values[idx - half * 2:idx - half]
    win2 = values[idx - half:idx]
    if len(win1) < 5 or len(win2) < 5:
        return 0

    combined = np.concatenate([win1, win2])
    bins = np.linspace(min(combined), max(combined), n_bins + 1)
    if bins[-1] == bins[0]:
        return 0

    h1, _ = np.histogram(win1, bins=bins, density=True)
    h2, _ = np.histogram(win2, bins=bins, density=True)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    kl = np.sum(h1 * (np.log(h1 + eps) - np.log(h2 + eps)))
    return abs(kl)


def feat_lumpiness(values: np.ndarray, idx: int, window_size: int, period: int) -> float:
    """Changing variance in remainder."""
    win = _rolling_window(values, idx, window_size)
    if len(win) < period * 2:
        return 0
    decomp = _stl_decompose(win, period)
    rem = decomp['remainder']
    chunks = []
    for i in range(0, len(rem), period):
        chunk = rem[i:i + period]
        if len(chunk) > 1:
            chunks.append(np.var(chunk))
    if len(chunks) < 2:
        return 0
    return np.var(chunks)


def feat_flatspots(values: np.ndarray, idx: int, window_size: int, n_intervals: int = 10) -> int:
    """Maximum run length within discretized buckets."""
    win = _rolling_window(values, idx, window_size)
    if len(win) < 2:
        return 0
    vmin, vmax = np.min(win), np.max(win)
    if vmax - vmin < 1e-10:
        return len(win)
    discretized = np.floor((win - vmin) / (vmax - vmin) * n_intervals).astype(int)
    discretized = np.clip(discretized, 0, n_intervals - 1)

    max_run = 1
    current_run = 1
    for i in range(1, len(discretized)):
        if discretized[i] == discretized[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def feat_diff_w(values: np.ndarray, idx: int, w: int) -> float:
    """Difference between current value and w-th previous value."""
    if idx < w:
        return 0
    return values[idx] - values[idx - w]


# =============================================================================
# Main Feature Extraction Pipeline
# =============================================================================

def extract_all_features(
    values: np.ndarray,
    labels: Optional[np.ndarray] = None,
    period: Optional[int] = None,
    wpr: int = 24
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Extract 37 features for each data point in a time series.

    Features:
    - Original value (1)
    - Statistical (11): Mean, Var, Crossingpoint, ACF1, ACFremainder, Trend,
      Linearity, Curvature, Entropy, ARCHtest.p, GARCHtest.p
    - Forecasting error (5×3=15): ME, RMSE, MAE, MPE, MAPE on 3 window sizes
    - Temporal (10): Max level shift, Max var shift, Max KL shift,
      Lumpiness, Flatspots, Diff-p/2, Diff-p, Diff-2p, Diff-wpr, Diff-wpr/2

    Returns: (feature_matrix, labels, feature_names)
    """
    n = len(values)

    if period is None:
        period = detect_period_fft(values)

    # Three window sizes for forecasting error (as fraction of period)
    p = max(period, 4)
    w1 = max(p // 4, 3)
    w2 = max(p // 2, 6)
    w3 = max(p, 12)

    # Main window size for statistical features
    ws = max(p, 10)

    feature_names = [
        'original_value',
        # Statistical
        'mean', 'var', 'crossingpoint', 'acf1', 'acfremainder',
        'trend', 'linearity', 'curvature', 'entropy', 'arch_test', 'garch_test',
        # Forecasting error (w1)
        'ME_w1', 'RMSE_w1', 'MAE_w1', 'MPE_w1', 'MAPE_w1',
        # Forecasting error (w2)
        'ME_w2', 'RMSE_w2', 'MAE_w2', 'MPE_w2', 'MAPE_w2',
        # Forecasting error (w3)
        'ME_w3', 'RMSE_w3', 'MAE_w3', 'MPE_w3', 'MAPE_w3',
        # Temporal
        'max_level_shift', 'max_var_shift', 'max_kl_shift',
        'lumpiness', 'flatspots',
        'diff_half_p', 'diff_p', 'diff_2p', 'diff_wpr', 'diff_half_wpr',
    ]

    features = np.zeros((n, len(feature_names)))

    for i in range(n):
        feat_idx = 0

        # Original value
        features[i, feat_idx] = values[i]
        feat_idx += 1

        # Statistical features
        features[i, feat_idx] = feat_mean(values, i, ws); feat_idx += 1
        features[i, feat_idx] = feat_var(values, i, ws); feat_idx += 1
        features[i, feat_idx] = feat_crossingpoint(values, i, ws); feat_idx += 1
        features[i, feat_idx] = feat_acf1(values, i, ws); feat_idx += 1
        features[i, feat_idx] = feat_acfremainder(values, i, ws, p); feat_idx += 1
        features[i, feat_idx] = feat_trend(values, i, ws, p); feat_idx += 1
        features[i, feat_idx] = feat_linearity(values, i, ws, p); feat_idx += 1
        features[i, feat_idx] = feat_curvature(values, i, ws, p); feat_idx += 1
        features[i, feat_idx] = feat_entropy(values, i, ws); feat_idx += 1
        features[i, feat_idx] = feat_arch_test(values, i, ws); feat_idx += 1
        features[i, feat_idx] = feat_garch_test(values, i, ws); feat_idx += 1

        # Forecasting error features
        for w in [w1, w2, w3]:
            err = _calc_forecast_error_features(values, i, w, p)
            for key in ['ME', 'RMSE', 'MAE', 'MPE', 'MAPE']:
                features[i, feat_idx] = err[key]
                feat_idx += 1

        # Temporal features
        features[i, feat_idx] = feat_max_level_shift(values, i, ws); feat_idx += 1
        features[i, feat_idx] = feat_max_var_shift(values, i, ws); feat_idx += 1
        features[i, feat_idx] = feat_max_kl_shift(values, i, ws); feat_idx += 1
        features[i, feat_idx] = feat_lumpiness(values, i, ws, p); feat_idx += 1
        features[i, feat_idx] = feat_flatspots(values, i, ws); feat_idx += 1
        features[i, feat_idx] = feat_diff_w(values, i, p // 2); feat_idx += 1
        features[i, feat_idx] = feat_diff_w(values, i, p); feat_idx += 1
        features[i, feat_idx] = feat_diff_w(values, i, p * 2); feat_idx += 1
        features[i, feat_idx] = feat_diff_w(values, i, wpr); feat_idx += 1
        features[i, feat_idx] = feat_diff_w(values, i, wpr // 2); feat_idx += 1

    # Handle NaN / inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, labels, feature_names


# =============================================================================
# CORAL Transformation
# =============================================================================

def coral_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    CORrelation ALignment (CORAL): align second-order statistics of
    source and target feature distributions.

    Returns: transformed source features aligned to target distribution.
    """
    # Center
    src_mean = np.mean(source, axis=0)
    tgt_mean = np.mean(target, axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    # Covariance matrices
    n_src = src_centered.shape[0]
    n_tgt = tgt_centered.shape[0]

    cov_src = (src_centered.T @ src_centered) / (n_src - 1) if n_src > 1 else np.cov(src_centered.T)
    cov_tgt = (tgt_centered.T @ tgt_centered) / (n_tgt - 1) if n_tgt > 1 else np.cov(tgt_centered.T)

    # Regularize
    eps = 1e-3
    d = source.shape[1]
    cov_src += eps * np.eye(d)
    cov_tgt += eps * np.eye(d)

    # Whitening: C_src^(-1/2)
    try:
        u_s, s_s, vt_s = np.linalg.svd(cov_src, hermitian=True)
        sqrt_s = np.sqrt(np.maximum(s_s, 1e-10))
        whitening = vt_s.T @ np.diag(1.0 / sqrt_s) @ u_s.T
    except:
        # Fallback
        whitening = np.linalg.pinv(cov_src) ** 0.5

    # Re-coloring: C_tgt^(1/2)
    try:
        u_t, s_t, vt_t = np.linalg.svd(cov_tgt, hermitian=True)
        sqrt_s = np.sqrt(np.maximum(s_t, 1e-10))
        recoloring = vt_t.T @ np.diag(sqrt_s) @ u_t.T
    except:
        recoloring = cov_tgt ** 0.5

    # Transform
    A = whitening @ recoloring
    transformed = src_centered @ A + tgt_mean

    return transformed


# =============================================================================
# ATAD Main Class
# =============================================================================

class ATAD:
    """
    Active Transfer Anomaly Detection.

    Parameters:
    - n_clusters: number of clusters for sub source domains (K)
    - n_rounds: number of active learning rounds (T)
    - samples_per_round: number of samples to label per round (d)
    - context_delta: context window radius for diversity
    - anomaly_threshold: probability threshold for anomaly decision
    """

    def __init__(
        self,
        n_clusters: int = 4,
        n_rounds: int = 3,
        samples_per_round: int = 60,
        context_delta: int = 10,
        anomaly_threshold: float = 0.5,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.n_rounds = n_rounds
        self.samples_per_round = samples_per_round
        self.context_delta = context_delta
        self.anomaly_threshold = anomaly_threshold
        self.random_state = random_state

        # Internal state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_centers_ = None
        self.models_ = []          # One RF per cluster
        self.feature_names_ = []
        self.period_ = None
        self._trained = False

    def fit_predict(
        self,
        src_values: np.ndarray,
        src_labels: np.ndarray,
        tgt_values: np.ndarray,
        return_probabilities: bool = False
    ) -> np.ndarray:
        """
        Train ATAD on source domain and predict on target domain.
        Uses transfer learning + active learning.

        Args:
            src_values: source domain time series values (labeled)
            src_labels: source domain labels (0=normal, 1=anomaly)
            tgt_values: target domain time series values (unlabeled)
            return_probabilities: if True, return anomaly probabilities

        Returns:
            Predicted labels for target domain
        """
        # 1. Extract features
        period = detect_period_fft(np.concatenate([src_values, tgt_values]))
        self.period_ = period

        src_feat, src_lbl, feat_names = extract_features_fast(
            src_values, src_labels, period, verbose=False
        )
        tgt_feat, _, _ = extract_features_fast(
            tgt_values, None, period, verbose=False
        )
        self.feature_names_ = feat_names

        # 2. Normalize
        X_src = self.scaler.fit_transform(src_feat)
        X_tgt = self.scaler.transform(tgt_feat)
        y_src = src_lbl.astype(int)

        # 3. K-means clustering on source
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        cluster_labels = self.kmeans.fit_predict(X_src)
        self.cluster_centers_ = self.kmeans.cluster_centers_

        # 4. Assign target samples to nearest cluster
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(X_tgt, self.cluster_centers_)
        tgt_assignments = np.argmin(dists, axis=1)

        # 5. For each cluster: CORAL + train base RF + active learning
        all_probs = np.zeros(len(tgt_values))

        for k in range(self.n_clusters):
            # Source samples in this cluster
            src_mask = cluster_labels == k
            X_src_k = X_src[src_mask]
            y_src_k = y_src[src_mask]

            # Target samples assigned to this cluster
            tgt_mask = tgt_assignments == k
            X_tgt_k = X_tgt[tgt_mask]
            tgt_indices = np.where(tgt_mask)[0]

            if len(X_src_k) < 10 or len(X_tgt_k) < 5:
                continue

            # CORAL transform source features to align with target
            X_src_transformed = coral_transform(X_src_k, X_tgt_k)

            # Train base Random Forest
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X_src_transformed, y_src_k)

            # Active Learning
            rf = self._active_learning(
                rf, X_tgt_k, tgt_indices, tgt_values
            )

            self.models_.append(rf)

            # Predict
            if len(X_tgt_k) > 0:
                probs = rf.predict_proba(X_tgt_k)[:, 1]
                all_probs[tgt_indices] = probs

        self._trained = True

        if return_probabilities:
            return all_probs

        return (all_probs >= self.anomaly_threshold).astype(int)

    def _active_learning(
        self,
        model: RandomForestClassifier,
        X_tgt: np.ndarray,
        tgt_indices: np.ndarray,
        tgt_values: np.ndarray
    ) -> RandomForestClassifier:
        """
        UCD (Uncertainty-Context Diversity) active learning.

        In each round, selects diverse & uncertain samples, assigns pseudo-labels,
        and retrains the model with the augmented dataset.
        """
        if len(X_tgt) < self.samples_per_round:
            return model

        current_model = model
        n_tgt = len(X_tgt)
        labeled_mask = np.zeros(n_tgt, dtype=bool)
        # Store pseudo-labels for all labeled points across rounds
        all_pseudo_labels = np.zeros(n_tgt, dtype=int)

        for round_idx in range(self.n_rounds):
            # Get probabilities
            probs = current_model.predict_proba(X_tgt)
            prob_normal = probs[:, 0]
            prob_anomaly = probs[:, 1]

            # Uncertainty = 1 - |P(normal) - P(anomaly)|
            uncertainty = 1.0 - np.abs(prob_normal - prob_anomaly)

            # Sort by uncertainty descending
            sorted_indices = np.argsort(uncertainty)[::-1]

            # UCD selection
            candidate_set = []

            for idx in sorted_indices:
                if labeled_mask[idx]:
                    continue

                if not candidate_set:
                    candidate_set.append(idx)
                    continue

                if len(candidate_set) >= self.samples_per_round:
                    break

                # Context diversity check
                is_in_context = False
                t = tgt_indices[idx]
                for c_idx in candidate_set:
                    ct = tgt_indices[c_idx]
                    if abs(t - ct) <= self.context_delta:
                        is_in_context = True
                        break

                if not is_in_context:
                    candidate_set.append(idx)

            # Simulate labeling with pseudo-labels
            if candidate_set:
                # Get pseudo-labels for newly selected candidates
                new_pseudo_labels = current_model.predict(X_tgt[candidate_set])

                for j, c in enumerate(candidate_set):
                    labeled_mask[c] = True
                    all_pseudo_labels[c] = new_pseudo_labels[j]

                # Build augmented dataset using ALL labeled points so far
                unlabeled_idx = np.where(~labeled_mask)[0]
                labeled_idx = np.where(labeled_mask)[0]

                X_augmented = np.vstack([
                    X_tgt[unlabeled_idx],
                    X_tgt[labeled_idx]
                ])
                y_augmented = np.concatenate([
                    np.zeros(len(unlabeled_idx)),
                    all_pseudo_labels[labeled_idx]
                ])

                current_model.fit(X_augmented, y_augmented)

        return current_model

    def predict(
        self,
        values: np.ndarray,
        return_probabilities: bool = False
    ) -> np.ndarray:
        """
        Predict anomalies on new time series data.
        Requires fit_predict to have been called first.
        """
        if not self._trained:
            raise RuntimeError("ATAD must be trained via fit_predict() first.")

        features, _, _ = extract_features_fast(values, None, self.period_, verbose=False)
        X = self.scaler.transform(features)

        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(X, self.cluster_centers_)
        assignments = np.argmin(dists, axis=1)

        all_probs = np.zeros(len(values))

        for k in range(self.n_clusters):
            mask = assignments == k
            if np.sum(mask) == 0:
                continue
            if k < len(self.models_):
                probs = self.models_[k].predict_proba(X[mask])[:, 1]
                all_probs[mask] = probs

        if return_probabilities:
            return all_probs
        return (all_probs >= self.anomaly_threshold).astype(int)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"F:\1")
    from data_loader import load_training_data, load_test_data
    from atad_features import extract_features_fast, detect_period_fft

    print("Loading data...")
    train = load_training_data(r"F:\1\jmeter_avg_elapsed_ms(1).csv")
    test = load_test_data(r"F:\1\jmeter_avg_elapsed_ms (1).csv")

    src_values = train['value'].values
    src_labels = train['label'].values
    tgt_values = test['value'].values

    print(f"Source: {len(src_values)} points, anomaly ratio: {src_labels.mean():.4f}")
    print(f"Target: {len(tgt_values)} points")

    print("\nTraining ATAD...")
    atad = ATAD(n_clusters=4, n_rounds=3, samples_per_round=30)
    preds = atad.fit_predict(src_values, src_labels, tgt_values)
    print(f"Predictions: {np.sum(preds)} anomalies out of {len(preds)} points")
    print(f"Anomaly ratio: {np.mean(preds):.4f}")
