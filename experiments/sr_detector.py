"""
Spectral Residual (SR) 异常检测算法
论文: Time-Series Anomaly Detection Service at Microsoft (KDD'19)

核心思想: 将视觉显著性检测中的Spectral Residual方法迁移到时序异常检测
- 对时序做FFT得到对数幅度谱
- 用均值滤波得到"平均谱"（背景）
- 残差 = 对数谱 - 平均谱（显著部分）
- 逆FFT得到显著图 → 异常分数
"""

import numpy as np
from scipy.fft import fft, ifft
from typing import Tuple


def spectral_residual(
    values: np.ndarray,
    window_size: int = 100,
    eval_window_size: int = 20,
    threshold_percentile: float = 98.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spectral Residual anomaly detection.

    Steps:
    1. Compute log amplitude spectrum via FFT
    2. Average filtering to get spectral average (background)
    3. Spectral residual = log spectrum - average spectrum
    4. Inverse FFT → saliency map in time domain
    5. Detect anomalies where saliency > threshold

    Args:
        values: time series values
        window_size: sliding window size for FFT
        eval_window_size: last N points in saliency map to evaluate
        threshold_percentile: percentile of saliency values as threshold

    Returns:
        anomaly_scores: saliency scores (higher = more anomalous)
        predictions: binary anomaly labels
    """
    n = len(values)
    anomaly_scores = np.zeros(n)
    predictions = np.zeros(n, dtype=int)

    # Pad window_size parameter
    ws = min(window_size, n // 2)

    for i in range(ws, n):
        # Extract current window
        window = values[i - ws:i]

        # 1. FFT → amplitude and phase
        fft_vals = fft(window)
        amplitude = np.abs(fft_vals)
        phase = np.angle(fft_vals)

        # 2. Log amplitude spectrum
        log_amplitude = np.log(amplitude + 1e-10)

        # 3. Average filtering (mean filter of size 3 in paper)
        avg_filter_size = 3
        kernel = np.ones(avg_filter_size) / avg_filter_size
        avg_log_amplitude = np.convolve(log_amplitude, kernel, mode='same')

        # 4. Spectral residual
        spectral_res = log_amplitude - avg_log_amplitude

        # 5. Inverse FFT → saliency map
        # Reconstruct complex spectrum with residual amplitude and original phase
        saliency_complex = np.exp(spectral_res + 1j * phase)
        saliency_map = np.abs(ifft(saliency_complex))

        # 6. Score for current point: use tail of saliency map
        eval_len = min(eval_window_size, ws)
        anomaly_scores[i] = np.max(saliency_map[-eval_len:])

    # Compute threshold from scores
    if np.max(anomaly_scores) > 1e-10:
        threshold = np.percentile(anomaly_scores[anomaly_scores > 1e-10], threshold_percentile)
    else:
        threshold = 0

    predictions = (anomaly_scores > threshold).astype(int)

    return anomaly_scores, predictions


# =============================================================================
# SR-based anomaly detector class
# =============================================================================

class SRDetector:
    """
    Spectral Residual anomaly detector for time series.
    """

    def __init__(
        self,
        window_size: int = 100,
        eval_window_size: int = 20,
        threshold_percentile: float = 98.0
    ):
        self.window_size = window_size
        self.eval_window_size = eval_window_size
        self.threshold_percentile = threshold_percentile
        self._threshold = None
        self._scores = None

    def fit_predict(
        self,
        values: np.ndarray,
        return_scores: bool = False
    ) -> np.ndarray:
        """
        Run SR detection on time series values.
        """
        scores, preds = spectral_residual(
            values,
            window_size=self.window_size,
            eval_window_size=self.eval_window_size,
            threshold_percentile=self.threshold_percentile
        )
        self._scores = scores
        self._threshold = np.percentile(scores[scores > 1e-10], self.threshold_percentile) if np.any(scores > 1e-10) else 0

        if return_scores:
            return scores
        return preds


# =============================================================================
# Holter-Winters Exponential Smoothing based detector (simple baseline)
# =============================================================================

def holt_winters_residual(
    values: np.ndarray,
    period: int = 24,
    alpha: float = 0.3,
    beta: float = 0.1,
    gamma: float = 0.1
) -> np.ndarray:
    """
    Holt-Winters triple exponential smoothing residuals.
    Returns the absolute residuals as anomaly scores.
    """
    n = len(values)
    if n < period * 2:
        return np.abs(values - np.mean(values))

    # Initialize
    level = np.mean(values[:period])
    trend = (np.mean(values[period:2*period]) - np.mean(values[:period])) / period
    seasonal = np.array([values[i] - level for i in range(period)])

    residuals = np.zeros(n)
    fitted = np.zeros(n)

    for i in range(n):
        s_idx = i % period
        if i < 2 * period:
            fitted[i] = level
        else:
            fitted[i] = level + trend + seasonal[s_idx]

        residuals[i] = abs(values[i] - fitted[i])

        # Update
        if i >= period:
            new_level = alpha * (values[i] - seasonal[s_idx]) + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            new_seasonal = gamma * (values[i] - new_level) + (1 - gamma) * seasonal[s_idx]

            level = new_level
            trend = new_trend
            seasonal[s_idx] = new_seasonal

    return residuals


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"F:\1")
    from data_loader import load_training_data, get_data_splits
    from evaluate import compute_metrics

    train = load_training_data(r"F:\1\jmeter_avg_elapsed_ms(1).csv")
    _, tgt_df = get_data_splits(train, test_ratio=0.5)
    tgt_vals = tgt_df['value'].values.astype(float)
    tgt_labels = tgt_df['label'].values.astype(int)

    # Test SR
    sr = SRDetector(window_size=100, threshold_percentile=95)
    preds = sr.fit_predict(tgt_vals)
    m = compute_metrics(tgt_labels, preds, use_adjustment=True)
    print(f"SR: F1={m['f1_score']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}")
