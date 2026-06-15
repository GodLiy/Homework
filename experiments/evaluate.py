"""
评估与可视化模块
Evaluation metrics and visualization for anomaly detection results.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Metric Calculation
# =============================================================================

def adjust_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_delay: int = 3
) -> np.ndarray:
    """
    Adjust predictions according to the strategy in FluxEV and ATAD papers:
    For a continuous anomaly segment, if detection occurs within max_delay
    points after the start, the entire segment is counted as TP.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        max_delay: allowed detection delay (points)

    Returns:
        adjusted y_pred
    """
    adjusted = y_pred.copy()
    n = len(y_true)

    # Find continuous anomaly segments in ground truth
    in_segment = False
    segment_start = 0

    for i in range(n):
        if y_true[i] == 1 and not in_segment:
            in_segment = True
            segment_start = i
        elif y_true[i] == 0 and in_segment:
            in_segment = False
            segment_end = i
            # Check if any detection in [segment_start, segment_end + max_delay]
            detect_window_end = min(segment_end + max_delay, n)
            detected = np.any(y_pred[segment_start:detect_window_end] == 1)
            if detected:
                adjusted[segment_start:segment_end] = 1
        # Handle segment at end of data
        if in_segment and i == n - 1:
            segment_end = n
            detected = np.any(y_pred[segment_start:segment_end] == 1)
            if detected:
                adjusted[segment_start:segment_end] = 1

    return adjusted


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    use_adjustment: bool = True,
    max_delay: int = 3
) -> Dict[str, float]:
    """
    Compute Precision, Recall, F1-Score, and additional metrics.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        use_adjustment: whether to use the delay adjustment strategy
        max_delay: allowed detection delay for adjustment

    Returns:
        Dictionary of metrics
    """
    if use_adjustment:
        y_pred_adj = adjust_predictions(y_true, y_pred, max_delay)
    else:
        y_pred_adj = y_pred

    tp = np.sum((y_pred_adj == 1) & (y_true == 1))
    fp = np.sum((y_pred_adj == 1) & (y_true == 0))
    fn = np.sum((y_pred_adj == 0) & (y_true == 1))
    tn = np.sum((y_pred_adj == 0) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Count continuous anomaly segments
    true_segments = count_anomaly_segments(y_true)
    pred_segments = count_anomaly_segments(y_pred)
    detected_segments = count_detected_segments(y_true, y_pred_adj)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'true_segments': true_segments,
        'pred_segments': pred_segments,
        'detected_segments': detected_segments
    }


def count_anomaly_segments(labels: np.ndarray) -> int:
    """Count number of continuous anomaly segments."""
    segments = 0
    in_segment = False
    for label in labels:
        if label == 1 and not in_segment:
            segments += 1
            in_segment = True
        elif label == 0:
            in_segment = False
    return segments


def count_detected_segments(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """Count how many true anomaly segments are detected."""
    detected = 0
    in_segment = False
    segment_detected = False
    n = len(y_true)

    for i in range(n):
        if y_true[i] == 1:
            if not in_segment:
                in_segment = True
                segment_detected = False
            if y_pred[i] == 1:
                segment_detected = True
        else:
            if in_segment and segment_detected:
                detected += 1
            in_segment = False
            segment_detected = False

    # Last segment
    if in_segment and segment_detected:
        detected += 1

    return detected


def print_metrics(metrics: Dict[str, float], title: str = "Results"):
    """Pretty print evaluation metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1-Score:   {metrics['f1_score']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Specificity:{metrics['specificity']:.4f}")
    print(f"  TP={metrics['tp']}, FP={metrics['fp']}, "
          f"FN={metrics['fn']}, TN={metrics['tn']}")
    print(f"  True anomaly segments: {metrics['true_segments']}")
    print(f"  Predicted segments:    {metrics['pred_segments']}")
    print(f"  Detected segments:     {metrics['detected_segments']}")
    print(f"{'=' * 60}")


# =============================================================================
# Visualization
# =============================================================================

def plot_results(
    timestamps: np.ndarray,
    values: np.ndarray,
    y_true: np.ndarray,
    y_pred: Dict[str, np.ndarray],
    scores: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None,
    title: str = "Anomaly Detection Results"
):
    """
    Plot time series with ground truth and predicted anomalies.

    Args:
        timestamps: datetime timestamps
        values: time series values
        y_true: ground truth labels
        y_pred: dict of method_name -> predicted labels
        scores: optional dict of method_name -> anomaly scores
        save_path: path to save the figure
        title: plot title
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("Warning: matplotlib not available for plotting.")
        return

    n_methods = len(y_pred)
    n_plots = 2 + n_methods + (1 if scores else 0)
    if scores:
        n_plots += len(scores) - 1  # Already added 1

    # Actually, let's simplify: value + true + one subplot per method
    method_names = list(y_pred.keys())
    n_plots = 2 + len(method_names) + (len(scores) if scores else 0)
    fig_height = 3 * n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(16, fig_height), sharex=True)

    # Plot 1: Original time series
    ax = axes[0]
    ax.plot(timestamps, values, 'b-', linewidth=0.8, alpha=0.7, label='Value')
    # Highlight true anomalies
    anomaly_mask = y_true == 1
    if np.any(anomaly_mask):
        ax.scatter(timestamps[anomaly_mask], values[anomaly_mask],
                   c='red', s=20, alpha=0.8, label='True Anomaly', zorder=5)
    ax.set_ylabel('Value')
    ax.set_title(f'{title} - Original Time Series with Ground Truth')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Ground truth only
    ax = axes[1]
    ax.fill_between(range(len(y_true)), 0, y_true, step='post',
                     color='red', alpha=0.4, label='True Anomalies')
    ax.set_ylabel('GT')
    ax.set_title('Ground Truth Anomaly Labels')
    ax.set_ylim(0, 1.2)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plot_idx = 2

    # Plot for each method: predictions
    for name in method_names:
        ax = axes[plot_idx]
        pred = y_pred[name]

        # Show correct vs incorrect predictions
        tp_mask = (pred == 1) & (y_true == 1)
        fp_mask = (pred == 1) & (y_true == 0)
        fn_mask = (pred == 0) & (y_true == 1)

        ax.plot(timestamps, values, 'gray', linewidth=0.5, alpha=0.4)
        if np.any(tp_mask):
            ax.scatter(timestamps[tp_mask], values[tp_mask],
                       c='green', s=20, alpha=0.8, label='TP', marker='o', zorder=5)
        if np.any(fp_mask):
            ax.scatter(timestamps[fp_mask], values[fp_mask],
                       c='orange', s=30, alpha=0.8, label='FP', marker='x', zorder=4)
        if np.any(fn_mask):
            ax.scatter(timestamps[fn_mask], values[fn_mask],
                       c='red', s=30, alpha=0.6, label='FN', marker='^', zorder=3)

        ax.set_ylabel('Value')
        ax.set_title(f'{name} - Predictions')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot scores if provided
    if scores:
        for name, score_arr in scores.items():
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.plot(timestamps, score_arr, 'purple', linewidth=0.8, alpha=0.7)
                ax.set_ylabel('Score')
                ax.set_title(f'{name} - Anomaly Score')
                ax.grid(True, alpha=0.3)
                plot_idx += 1

    # Format x-axis
    if len(timestamps) > 0:
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.xlabel('Timestamp')
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.savefig(r"F:\1\results.png", dpi=150, bbox_inches='tight')
        print(f"Figure saved to: F:\\1\\results.png")

    plt.close()


# =============================================================================
# Comparison Table
# =============================================================================

def generate_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Generate a comparison table from results dictionary.

    Args:
        results: dict of method_name -> metrics dict

    Returns:
        DataFrame with comparison table
    """
    rows = []
    for method, metrics in results.items():
        row = {
            'Method': method,
            'Precision': f"{metrics.get('precision', 0):.4f}",
            'Recall': f"{metrics.get('recall', 0):.4f}",
            'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
            'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'TP': metrics.get('tp', 0),
            'FP': metrics.get('fp', 0),
            'FN': metrics.get('fn', 0),
            'TN': metrics.get('tn', 0),
            'Detected/Total Segments': f"{metrics.get('detected_segments', 0)}/{metrics.get('true_segments', 0)}"
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # Quick test
    y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 0])

    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Metrics")
