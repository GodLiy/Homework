"""
FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection
论文: WSDM'21 - FluxEV

核心组件:
1. 数据预处理: 缺失值填充
2. 波动提取: EWMA预测 → 计算预测误差
3. 两步平滑:
   - 第一步: 序列平滑 (消除局部噪声)
   - 第二步: 周期平滑 (消除周期模式噪声)
4. 自动阈值: SPOT + MOM (Method of Moments) 优化
5. 流式检测
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import deque
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Data Preprocessing
# =============================================================================

def preprocess_fill_missing(
    values: np.ndarray,
    period: int,
    short_threshold: int = 5
) -> np.ndarray:
    """
    FluxEV missing value filling strategy:
    (1) Short gaps (< short_threshold points): linear interpolation
    (2) Long gaps: same time slot from previous period + bias (half of mean diff)
    """
    result = values.copy().astype(float)
    n = len(result)

    # Identify NaN positions
    nan_mask = np.isnan(result)
    if not np.any(nan_mask):
        return result

    # Process contiguous gaps
    i = 0
    while i < n:
        if nan_mask[i]:
            gap_start = i
            while i < n and nan_mask[i]:
                i += 1
            gap_end = i
            gap_len = gap_end - gap_start

            if gap_len < short_threshold:
                # Short gap: linear interpolation
                before = result[gap_start - 1] if gap_start > 0 else None
                after = result[gap_end] if gap_end < n else None

                if before is not None and after is not None:
                    interpolated = np.linspace(before, after, gap_len + 2)[1:-1]
                    result[gap_start:gap_end] = interpolated
                elif before is not None:
                    result[gap_start:gap_end] = before
                elif after is not None:
                    result[gap_start:gap_end] = after
                else:
                    result[gap_start:gap_end] = 0
            else:
                # Long gap: use previous period + bias
                for j in range(gap_start, gap_end):
                    p_idx = j - period
                    if p_idx >= 0 and not np.isnan(result[p_idx]):
                        # Calculate bias
                        bias = 0
                        if j >= period * 2:
                            prev_period_vals = result[j - 2*period:j - period]
                            curr_period_vals = result[j - period:j]
                            prev_mean = np.nanmean(prev_period_vals)
                            curr_mean = np.nanmean(curr_period_vals)
                            if not (np.isnan(prev_mean) or np.isnan(curr_mean)):
                                bias = (curr_mean - prev_mean) / 2
                        result[j] = result[p_idx] + bias
                    else:
                        result[j] = np.nanmean(result[max(0, j - 10):j])
        else:
            i += 1

    # Final fallback: fill any remaining NaN with forward fill
    mask = np.isnan(result)
    if np.any(mask):
        # forward fill
        for i in range(1, n):
            if np.isnan(result[i]):
                result[i] = result[i - 1] if not np.isnan(result[i - 1]) else 0
        # backward fill for leading NaN
        for i in range(n - 2, -1, -1):
            if np.isnan(result[i]):
                result[i] = result[i + 1] if not np.isnan(result[i + 1]) else 0

    return result


# =============================================================================
# Fluctuation Extraction (EWMA)
# =============================================================================

def ewma_predict(
    values: np.ndarray,
    idx: int,
    window_size: int = 10,
    alpha: float = 0.3
) -> float:
    """
    Exponentially Weighted Moving Average prediction.
    EWMA = (X_{t-1} + (1-α)X_{t-2} + ... + (1-α)^{s-1}X_{t-s}) /
           (1 + (1-α) + ... + (1-α)^{s-1})
    """
    s = min(window_size, idx)
    if s == 0:
        return values[idx] if idx >= 0 else 0

    start = max(0, idx - s)
    window = values[start:idx]
    if len(window) == 0:
        return values[idx] if idx >= 0 else 0

    s = len(window)
    weights = np.array([(1 - alpha) ** (s - 1 - i) for i in range(s)])
    weight_sum = np.sum(weights)

    if weight_sum < 1e-10:
        return np.mean(window)

    return np.sum(window * weights) / weight_sum


def extract_fluctuations(
    values: np.ndarray,
    window_size: int = 10,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Extract local fluctuations using EWMA.
    E_i = X_i - EWMA(X_{i-s, i-1})
    """
    n = len(values)
    E = np.zeros(n)

    for i in range(n):
        if i >= window_size:
            pred = ewma_predict(values, i, window_size, alpha)
            E[i] = values[i] - pred
        else:
            E[i] = 0  # Not enough history

    return E


# =============================================================================
# Two-Step Smoothing
# =============================================================================

def _std(values: np.ndarray) -> float:
    """Standard deviation of array."""
    if len(values) < 2:
        return 0
    return np.std(values, ddof=1)


def _max_in_window(arr: np.ndarray, center: int, half_d: int) -> float:
    """Max value in window [center - half_d, center + half_d]."""
    start = max(0, center - half_d)
    end = min(len(arr), center + half_d + 1)
    if start >= end:
        return 0
    return np.max(arr[start:end])


def first_step_smoothing(E: np.ndarray, s: int = 10) -> np.ndarray:
    """
    First-step smoothing: sequential processing to eliminate local noises.

    Δ = σ(E_{i-s, i}) - σ(E_{i-s, i-1})
    F_i = max(Δ, 0)

    Rationale: if adding E_i causes a large increase in std dev of the window,
    it is a potential anomaly; otherwise, its fluctuation is normal noise.
    """
    n = len(E)
    F = np.zeros(n)

    for i in range(2 * s, n):
        window_full = E[i - s:i + 1]
        window_prev = E[i - s:i]

        sigma_full = _std(window_full)
        sigma_prev = _std(window_prev)

        delta = sigma_full - sigma_prev
        F[i] = max(delta, 0)

    return F


def second_step_smoothing(
    F: np.ndarray,
    period: int,
    p: int = 5,
    d: int = 2,
    n_none: List = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Second-step smoothing: periodic processing to eliminate periodic patterns.

    M_{i-d} = max(F_{i-2d, i})
    F'_i = F_i - max(M_{i-l*(p-1)}, ..., M_{i-2l}, M_{i-l})
    S_i = max(F'_i, 0)

    Uses a local max window (±d) to handle data drift in periodic patterns.
    Returns: (S, M, F_prime) where S is the final smoothed fluctuation.
    """
    n = len(F)
    l = period  # period length in points
    M = np.zeros(n)
    F_prime = np.zeros(n)
    S = np.zeros(n)

    # Start index: need 2s + d + l*(p-1) as warm-up
    start_idx = 2 * s_val + 2 * d + l * (p - 1)
    if start_idx >= n:
        start_idx = max(2 * d, 0)

    # Check if specific s was used (we'll infer from F)
    s_val_actual = 10  # default

    for i in range(start_idx, n):
        # M_{i-d}: local max of F in window [i-2d, i]
        center_m = i - d
        M_val = _max_in_window(F, center_m, d)

        # If F was set to None (for previously detected anomalies), skip
        if n_none is not None and i in n_none:
            M[center_m] = M_val
            continue

        M[center_m] = M_val

        # Periodic smoothing: subtract max of past p periods' M values
        past_max = 0
        for k in range(1, p + 1):
            past_idx = i - l * k - d
            if past_idx >= start_idx - 2 * d:
                past_max = max(past_max, M[past_idx])

        F_prime[i] = F[i] - past_max
        S[i] = max(F_prime[i], 0)

    return S, M, F_prime


# Global variable for s used in smoothing functions
s_val = 10


def ext_and_smooth(
    X: np.ndarray,
    s: int = 10,
    p: int = 5,
    d: int = 2,
    period: int = 24,
    alpha: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full fluctuation extraction and smoothing pipeline (Algorithm 1 in FluxEV).

    Returns: (E, F, S, M)
      E: original extracted fluctuations
      F: after first-step smoothing
      S: after second-step smoothing (final anomaly score)
      M: local max values used in periodic smoothing
    """
    global s_val
    s_val = s
    n = len(X)

    # Initialize
    E = np.zeros(n)
    F = np.zeros(n)
    S = np.zeros(n)
    M = np.zeros(n)

    for i in range(n):
        # Step 1: EWMA fluctuation extraction (after s points)
        if i > s:
            pred = ewma_predict(X, i, s, alpha)
            E[i] = X[i] - pred
            # Use EWMA predicted value approach instead
            # Actually using direct EWMA formula
            vals = X[max(0, i - s):i]
            if len(vals) > 0:
                weights = np.array([(1 - alpha) ** (len(vals) - 1 - j) for j in range(len(vals))])
                weight_sum = np.sum(weights)
                if weight_sum > 1e-10:
                    pred2 = np.sum(vals * weights) / weight_sum
                    E[i] = X[i] - pred2

        # Step 2: First-step smoothing (after 2s points)
        if i > 2 * s:
            window_full = E[i - s:i + 1]
            window_prev = E[i - s:i]
            sigma_full = _std(window_full)
            sigma_prev = _std(window_prev)
            delta = sigma_full - sigma_prev
            F[i] = max(delta, 0)

        # Step 3: M update (after 2s + 2d points)
        if i > 2 * s + 2 * d:
            center_m = i - d
            M[center_m] = _max_in_window(F, center_m, d)

        # Step 4: Second-step smoothing (after 2s + d + period*(p-1) points)
        if i > 2 * s + d + period * (p - 1):
            past_max_vals = []
            for k in range(1, p + 1):
                past_idx = i - period * k - d
                if past_idx >= 0:
                    past_max_vals.append(M[past_idx])
            past_max = max(past_max_vals) if past_max_vals else 0

            F_prime_i = F[i] - past_max
            S[i] = max(F_prime_i, 0)

    return E, F, S, M


# =============================================================================
# SPOT with MOM (Method of Moments)
# =============================================================================

def mom_estimate_gpd(peaks: np.ndarray) -> Tuple[float, float]:
    """
    Estimate GPD parameters using Method of Moments (MOM).

    For GPD: E(Y) = σ / (1 - γ), Var(Y) = σ² / ((1 - γ)² * (1 - 2γ))

    MOM estimates:
      γ̂ = (1 - μ²/σ²_sample) / 2
      σ̂ = μ * (1 - γ̂)

    where μ is sample mean of peaks, σ²_sample is sample variance of peaks.
    """
    if len(peaks) < 3:
        return 0.0, 0.1  # Default small values

    mu = np.mean(peaks)
    sigma2 = np.var(peaks, ddof=1)

    if sigma2 < 1e-10 or mu < 1e-10:
        return 0.0, max(mu, 0.1)

    # γ̂ = (1 - μ²/σ²) / 2
    gamma_hat = (1.0 - mu ** 2 / sigma2) / 2.0

    # Clamp γ to valid range for GPD: γ < 0.5 (for finite variance), γ > -1
    gamma_hat = max(-0.99, min(0.49, gamma_hat))

    # σ̂ = μ * (1 - γ̂)
    sigma_hat = mu * (1.0 - gamma_hat)
    sigma_hat = max(sigma_hat, 0.01)

    return gamma_hat, sigma_hat


def pot_init(
    values: np.ndarray,
    q: float = 1e-3,
    init_quantile: float = 0.98
) -> Dict:
    """
    Initialize POT (Peaks-over-Threshold) using MOM.
    This is the initialization step of SPOT.

    Parameters:
    - values: initial data to compute threshold from
    - q: risk coefficient (controls false positive rate)
    - init_quantile: quantile for initial threshold t
    """
    n = len(values)
    t = float(np.quantile(values, init_quantile))

    # Find peaks (values above initial threshold)
    peaks = values[values > t] - t  # excesses over threshold
    Nt = len(peaks)

    if Nt < 3:
        return {'t': t, 'thF': float(np.max(values) * 1.5),
                'gamma': 0.0, 'sigma': 0.1, 'Nt': Nt, 'n': n}

    # Estimate GPD parameters using MOM
    gamma, sigma = mom_estimate_gpd(peaks)

    # Calculate final threshold
    thF = calc_threshold(q, gamma, sigma, n, Nt, t)

    return {
        't': t,
        'thF': thF,
        'gamma': gamma,
        'sigma': sigma,
        'Nt': Nt,
        'n': n
    }


def calc_threshold(
    q: float,
    gamma: float,
    sigma: float,
    n: int,
    Nt: int,
    t: float
) -> float:
    """
    Calculate the anomaly threshold thF.

    thF = t + (σ̂ / γ̂) * ((q * n / Nt)^{-γ̂} - 1)

    If γ̂ ≈ 0, use limit: thF = t + σ̂ * log(q * n / Nt)
    """
    if Nt == 0:
        return t * 1.5 if t > 0 else 1.0

    ratio = (q * n) / Nt

    if abs(gamma) < 1e-6:
        # Limit as γ → 0
        thF = t + sigma * np.log(ratio)
    else:
        try:
            thF = t + (sigma / gamma) * (ratio ** (-gamma) - 1)
        except (OverflowError, ValueError):
            thF = t + sigma * 10  # fallback

    return thF


def spot_update(
    state: Dict,
    new_value: float,
    q: float
) -> Tuple[bool, Dict]:
    """
    Update SPOT with a new streaming value.
    Returns (is_anomaly, updated_state).
    """
    thF = state['thF']
    t = state['t']

    if new_value > thF:
        # Anomaly detected
        return True, state

    # Update
    state['n'] += 1

    if new_value > t:
        # This is a new peak
        excess = new_value - t
        # We need to maintain peaks list; use incremental approach
        # Since storing full list is expensive, use approximate update
        state['Nt'] = state.get('Nt', 0) + 1

        # Recalculate GPD parameters (approximate)
        Nt = state['Nt']

        if Nt >= 3:
            # Approximate: keep running mean and variance of peaks
            old_mean = state.get('peak_mean', excess)
            old_var = state.get('peak_var', 0)

            new_mean = old_mean + (excess - old_mean) / Nt
            if Nt > 1:
                new_var = ((Nt - 2) * old_var + (excess - new_mean) * (excess - old_mean)) / (Nt - 1)
            else:
                new_var = 0

            state['peak_mean'] = new_mean
            state['peak_var'] = new_var

            gamma, sigma = mom_estimate_gpd(np.array([new_mean]))  # approximate
            # Use running stats to estimate
            if new_var > 1e-10 and new_mean > 1e-10:
                gamma = (1.0 - new_mean ** 2 / new_var) / 2.0
                gamma = max(-0.99, min(0.49, gamma))
                sigma = new_mean * (1.0 - gamma)
                sigma = max(sigma, 0.01)

            state['gamma'] = gamma
            state['sigma'] = sigma
            state['thF'] = calc_threshold(q, gamma, sigma, state['n'], Nt, t)

    return False, state


# =============================================================================
# FluxEV Main Class
# =============================================================================

class FluxEV:
    """
    FluxEV: Fast and Effective Unsupervised Anomaly Detection Framework.

    Parameters:
    - s: EWMA window size (default 10)
    - p: number of past periods for periodic smoothing (default 5)
    - d: half window for data drift handling (default 2)
    - alpha: EWMA smoothing factor (default 0.3)
    - q: risk coefficient for SPOT (default 1e-4)
    - init_points: number of points to initialize SPOT (default 1000)
    - period: period length in points (auto-detected if None)
    """

    def __init__(
        self,
        s: int = 10,
        p: int = 5,
        d: int = 2,
        alpha: float = 0.3,
        q: float = 1e-4,
        init_points: int = 1000,
        period: Optional[int] = None
    ):
        self.s = s
        self.p_periods = p
        self.d = d
        self.alpha = alpha
        self.q = q
        self.init_points = init_points
        self.period = period

        # Internal state
        self._spot_state = None
        self._trained = False
        self._warmup = 0
        self._E_history = deque(maxlen=10000)
        self._M_history = deque(maxlen=10000)
        self._detected_period = None

    def fit(self, values: np.ndarray):
        """
        Initialize FluxEV on a training portion of data.
        This performs:
        1. Period detection
        2. Feature extraction & smoothing pipeline
        3. SPOT initialization
        """
        n = len(values)

        # Detect period if not provided
        if self.period is None:
            self.period = self._detect_period(values)
        self._detected_period = self.period

        # Warm-up points needed
        self._warmup = 2 * self.s + self.d + self.period * (self.p_periods - 1)

        # Extract and smooth
        E, F, S, M = ext_and_smooth(
            values, self.s, self.p_periods, self.d, self.period, self.alpha
        )

        # Use valid S values (after warm-up) for SPOT initialization
        valid_start = min(self._warmup, n - self.init_points)
        init_values = S[valid_start:min(valid_start + self.init_points, n)]

        if len(init_values) < 100:
            init_values = S[max(0, n - 500):n]
        if len(init_values) < 10:
            init_values = np.abs(S)  # fallback

        self._spot_state = pot_init(init_values, self.q)

        # Store history for streaming
        self._E_history.clear()
        self._M_history.clear()
        for i in range(n):
            self._E_history.append(E[i])
            self._M_history.append(M[i] if i < len(M) else 0)

        self._trained = True
        return self

    def _detect_period(self, values: np.ndarray) -> int:
        """Detect dominant period using FFT."""
        n = len(values)
        if n < 16:
            return max(n // 2, 1)

        detrended = values - np.mean(values)
        fft_vals = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(n)
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        mag = np.abs(fft_vals[pos_mask])

        if len(mag) > 1:
            dom = np.argmax(mag[1:]) + 1
            if freqs[dom] > 0:
                p = int(1.0 / freqs[dom])
                return max(2, min(p, n // 3))
        return max(n // 10, 2)

    def predict(
        self,
        values: np.ndarray,
        return_scores: bool = False
    ) -> np.ndarray:
        """
        Predict anomalies in streaming fashion.

        Args:
            values: time series values to detect anomalies on
            return_scores: if True, return S scores instead of binary labels

        Returns:
            Binary anomaly labels (or S scores if return_scores=True)
        """
        if not self._trained:
            raise RuntimeError("FluxEV must be fit() first.")

        n = len(values)
        predictions = np.zeros(n, dtype=int)
        scores = np.zeros(n)

        # Convert deques to lists for indexing
        E_list = list(self._E_history)
        M_list = list(self._M_history)
        F_history = []  # Track F values for smoothing

        state = dict(self._spot_state)

        for i in range(n):
            # EWMA prediction
            if i + len(E_list) >= self.s:
                hist_start = max(0, i + len(E_list) - self.s)
                hist_values = values[:i + 1]
                vals = values[max(0, i - self.s):i]
                if len(vals) > 0:
                    weights = np.array([(1 - self.alpha) ** (len(vals) - 1 - j)
                                      for j in range(len(vals))])
                    w_sum = np.sum(weights)
                    if w_sum > 1e-10:
                        pred = np.sum(vals * weights) / w_sum
                    else:
                        pred = np.mean(vals)
                else:
                    pred = values[i]
                E_i = values[i] - pred
            else:
                E_i = 0

            E_list.append(E_i)

            # First-step smoothing
            total_len = i + 1 + len(E_list) - n
            if total_len > 2 * self.s:
                s_idx = total_len  # current index in accumulated E
                window_full = np.array(E_list[s_idx - self.s:s_idx + 1])
                window_prev = np.array(E_list[s_idx - self.s:s_idx])
                sigma_full = _std(window_full)
                sigma_prev = _std(window_prev)
                F_i = max(sigma_full - sigma_prev, 0)
            else:
                F_i = 0

            F_history.append(F_i)

            # M update
            if total_len > 2 * self.s + 2 * self.d:
                center_m = total_len - self.d
                M_list.append(_max_in_window(np.array(F_history), center_m, self.d))
            else:
                M_list.append(0)

            # Second-step smoothing
            if total_len > 2 * self.s + self.d + self.period * (self.p_periods - 1):
                past_max_vals = []
                for k in range(1, self.p_periods + 1):
                    past_idx = total_len - self.period * k - self.d
                    if 0 <= past_idx < len(M_list):
                        past_max_vals.append(M_list[past_idx])
                past_max = max(past_max_vals) if past_max_vals else 0

                S_i = max(F_i - past_max, 0)
            else:
                S_i = 0

            scores[i] = S_i

            # SPOT detection
            is_anomaly, state = spot_update(state, S_i, self.q)
            predictions[i] = 1 if is_anomaly else 0

            # If anomaly detected, set F_i = None for reference
            if is_anomaly:
                # Mark as None (0 in our impl) to avoid using as reference
                F_history[-1] = 0

        # Update internal state
        self._E_history = deque(E_list[-10000:], maxlen=10000)
        self._M_history = deque(M_list[-10000:], maxlen=10000)
        self._spot_state = state

        if return_scores:
            return scores
        return predictions

    def fit_predict(
        self,
        train_values: np.ndarray,
        test_values: np.ndarray,
        return_scores: bool = False
    ) -> np.ndarray:
        """
        Convenience method: fit on train, predict on test.
        """
        self.fit(train_values)
        return self.predict(test_values, return_scores)


# =============================================================================
# Simplified FluxEV (non-streaming, for batch evaluation)
# =============================================================================

class FluxEVBatch:
    """
    Batch version of FluxEV for offline evaluation.
    Performs full computation at once instead of streaming.
    """

    def __init__(
        self,
        s: int = 10,
        p: int = 5,
        d: int = 2,
        alpha: float = 0.3,
        q: float = 1e-4,
        init_quantile: float = 0.98
    ):
        self.s = s
        self.p_periods = p
        self.d = d
        self.alpha = alpha
        self.q = q
        self.init_quantile = init_quantile
        self.period = None
        self._threshold = None

    def fit_predict(
        self,
        values: np.ndarray,
        return_scores: bool = False
    ) -> np.ndarray:
        """
        Full FluxEV pipeline on a single time series.
        """
        n = len(values)

        # Detect period, but cap to reasonable fraction of data length
        if self.period is None:
            raw_period = self._detect_period(values)
            # Cap period to ensure warmup < 50% of data
            max_period = max(4, (n // 2 - 2 * self.s - self.d) // max(self.p_periods - 1, 1))
            self.period = min(raw_period, max_period)
            self.period = max(4, self.period)

        # Dynamically reduce p if period is still too large
        p_actual = self.p_periods
        warmup = 2 * self.s + self.d + self.period * (p_actual - 1)
        while warmup > n // 2 and p_actual > 1:
            p_actual -= 1
            warmup = 2 * self.s + self.d + self.period * (p_actual - 1)

        # Fill missing values
        values_clean = preprocess_fill_missing(values, self.period)

        # Extract fluctuations and smooth
        E, F, S, M = ext_and_smooth(
            values_clean, self.s, p_actual, self.d,
            self.period, self.alpha
        )

        # Initialize SPOT on S values after warm-up
        warmup = min(warmup, n // 4)

        init_values = S[warmup:min(warmup + 1000, n)]
        if len(init_values) < 100:
            # Use all S values excluding leading zeros
            nonzero_s = S[S > 0]
            if len(nonzero_s) > 100:
                init_values = nonzero_s
            else:
                init_values = S[max(0, n - 500):n]
        if len(init_values) < 10:
            # Fallback: use raw values for threshold
            init_values = np.abs(values_clean)

        spot_state = pot_init(init_values, self.q, self.init_quantile)
        self._threshold = spot_state['thF']

        # Apply threshold to all S values
        predictions = np.zeros(n, dtype=int)
        for i in range(n):
            if S[i] > spot_state['thF']:
                predictions[i] = 1

        if return_scores:
            return S
        return predictions

    def _detect_period(self, values: np.ndarray) -> int:
        n = len(values)
        if n < 16:
            return max(n // 2, 1)
        detrended = values - np.mean(values)
        fft_vals = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(n)
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        mag = np.abs(fft_vals[pos_mask])
        if len(mag) > 1:
            dom = np.argmax(mag[1:]) + 1
            if freqs[dom] > 0:
                p = int(1.0 / freqs[dom])
                return max(2, min(p, n // 3))
        return max(n // 10, 2)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"F:\1")
    from data_loader import load_training_data

    print("Loading data...")
    train = load_training_data(r"F:\1\jmeter_avg_elapsed_ms(1).csv")
    values = train['value'].values
    labels = train['label'].values

    print(f"Data: {len(values)} points, anomaly ratio: {labels.mean():.4f}")

    print("\nRunning FluxEV batch...")
    fluxev = FluxEVBatch(s=10, p=5, d=2, q=1e-4)
    preds = fluxev.fit_predict(values)
    print(f"Threshold: {fluxev._threshold:.4f}")
    print(f"Predictions: {np.sum(preds)} anomalies out of {len(preds)}")
    print(f"Anomaly ratio: {np.mean(preds):.4f}")

    # Quick evaluation
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
