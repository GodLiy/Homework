"""
数据加载与预处理模块
Data loading and preprocessing for JMeter time series data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def load_training_data(filepath: str) -> pd.DataFrame:
    """
    Load labeled training data (with 'label' column).
    - label=0: normal
    - label=1: anomaly
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_test_data(filepath: str, label_filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load unlabeled test data (without 'label' column).
    If label_filepath is provided (same as training data for evaluation),
    labels are merged for ground truth comparison.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('timestamp').reset_index(drop=True)

    if label_filepath:
        label_df = pd.read_csv(label_filepath)
        label_df.columns = label_df.columns.str.strip().str.lower()
        label_df['timestamp'] = pd.to_datetime(label_df['timestamp'], unit='s')
        df = df.merge(label_df[['timestamp', 'label']], on='timestamp', how='left')

    return df


def normalize_series(values: np.ndarray) -> np.ndarray:
    """Min-max normalization."""
    vmin, vmax = np.min(values), np.max(values)
    if vmax - vmin < 1e-8:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def detect_period(values: np.ndarray, sample_rate: int = 5) -> int:
    """
    Detect dominant period using FFT (Discrete Fourier Transform).
    sample_rate: sampling interval in seconds (default 5 for JMeter 5s interval).
    Returns period length in number of data points.
    """
    n = len(values)
    if n < 16:
        return max(n // 2, 1)

    fft_vals = np.fft.fft(values - np.mean(values))
    freqs = np.fft.fftfreq(n)
    # Only consider positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    magnitudes = np.abs(fft_vals[pos_mask])

    # Find the dominant frequency
    if len(magnitudes) > 0:
        dominant_idx = np.argmax(magnitudes[1:]) + 1  # skip DC (idx 0)
        dominant_freq = freqs[dominant_idx]
        if dominant_freq > 0:
            period = int(1.0 / dominant_freq)
            # Clamp to reasonable range for JMeter data
            # 5-second intervals: 12 points = 1 min, 720 points = 1 hour, 17280 = 1 day
            period = max(2, min(period, n // 3))
            return period
    return max(n // 10, 2)


def get_data_splits(df: pd.DataFrame, test_ratio: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test by temporal order.
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def fill_missing_values(values: np.ndarray, period: int) -> np.ndarray:
    """
    Fill missing values using FluxEV's strategy:
    - Short gaps (<5 points): linear interpolation
    - Long gaps (>=5 points): use same time slot from previous period + bias
    """
    result = values.copy().astype(float)
    n = len(result)
    nan_mask = np.isnan(result)

    if not np.any(nan_mask):
        return result

    # Find contiguous NaN segments
    in_gap = False
    gap_start = 0
    for i in range(n):
        if nan_mask[i] and not in_gap:
            in_gap = True
            gap_start = i
        elif not nan_mask[i] and in_gap:
            in_gap = False
            gap_end = i
            gap_len = gap_end - gap_start

            if gap_len < 5:
                # Linear interpolation
                if gap_start > 0 and gap_end < n:
                    result[gap_start:gap_end] = np.linspace(
                        result[gap_start - 1], result[gap_end], gap_len + 2
                    )[1:-1]
                elif gap_start == 0:
                    result[gap_start:gap_end] = result[gap_end]
                else:
                    result[gap_start:gap_end] = result[gap_start - 1]
            else:
                # Use previous period values + bias
                for j in range(gap_start, gap_end):
                    p_idx = j - period
                    if p_idx >= 0 and not np.isnan(result[p_idx]):
                        # Bias: half the difference between means of two periods
                        period_diff = 0
                        if j >= period * 2:
                            prev_period_mean = np.nanmean(result[j - 2 * period:j - period])
                            curr_period_mean = np.nanmean(result[j - period:j])
                            if not (np.isnan(prev_period_mean) or np.isnan(curr_period_mean)):
                                period_diff = (curr_period_mean - prev_period_mean) / 2
                        result[j] = result[p_idx] + period_diff
                    else:
                        result[j] = np.nanmean(result[max(0, j - 10):j])

        if in_gap and i == n - 1:
            gap_end = n
            gap_len = gap_end - gap_start
            if gap_len < 5:
                if gap_start > 0:
                    result[gap_start:gap_end] = result[gap_start - 1]
                else:
                    result[gap_start:gap_end] = 0
            else:
                for j in range(gap_start, gap_end):
                    p_idx = j - period
                    if p_idx >= 0 and not np.isnan(result[p_idx]):
                        result[j] = result[p_idx]
                    else:
                        result[j] = 0

    return result


if __name__ == "__main__":
    # Quick test
    train = load_training_data(r"F:\1\jmeter_avg_elapsed_ms(1).csv")
    test = load_test_data(r"F:\1\jmeter_avg_elapsed_ms (1).csv")
    print(f"Training data: {train.shape}")
    print(f"Test data: {test.shape}")
    print(f"Training columns: {train.columns.tolist()}")
    print(f"Test columns: {test.columns.tolist()}")
    print(f"Anomaly ratio in training: {train['label'].mean():.4f}")
    period = detect_period(train['value'].values)
    print(f"Detected period: {period} points")
