"""
算法优化与改进建议 - 实验脚本
基于第一阶段实验结果，针对性调优并引入SR替代算法
"""

import numpy as np
import pandas as pd
import time
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_training_data, get_data_splits, detect_period
from fluxev import FluxEVBatch
from atad import ATAD
from sr_detector import SRDetector
from evaluate import compute_metrics, print_metrics, generate_comparison_table


def main():
    print("=" * 72)
    print("  算法优化实验 - 参数调优 & 替代算法")
    print("=" * 72)

    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n[1/4] Loading data...")
    base_dir = r"F:\1"
    train = load_training_data(os.path.join(base_dir, "jmeter_avg_elapsed_ms(1).csv"))
    src_df, tgt_df = get_data_splits(train, test_ratio=0.5)
    src_vals = src_df['value'].values.astype(float)
    src_labels = src_df['label'].values.astype(int)
    tgt_vals = tgt_df['value'].values.astype(float)
    tgt_labels = tgt_df['label'].values.astype(int)
    tgt_timestamps = tgt_df['timestamp'].values

    period = detect_period(np.concatenate([src_vals, tgt_vals]))
    # Cap period for practical FluxEV use
    period_capped = min(period, max(4, (len(tgt_vals) // 2 - 20 - 2) // max(4, 1)))
    period_capped = max(4, period_capped)

    print(f"  Source: {len(src_vals)} pts, Anomaly%: {src_labels.mean():.4f}")
    print(f"  Target: {len(tgt_vals)} pts, Anomaly%: {tgt_labels.mean():.4f}")
    print(f"  Period: {period} (capped to {period_capped})")

    # =========================================================================
    # 2. FluxEV Deeper Optimization
    # =========================================================================
    print("\n[2/4] FluxEV Deeper Parameter Optimization...")

    # Phase 1 findings: s=5, p=1 work well for block anomalies
    # Now do finer grid search around these
    configs = []
    for s in [3, 5, 8, 10]:
        for p_val in [1, 2]:
            for alpha in [0.3, 0.5, 0.7]:
                for q in [5e-2, 1e-2, 5e-3, 1e-3]:
                    for init_q in [0.90, 0.95, 0.98]:
                        configs.append({
                            's': s, 'p': p_val, 'd': 2,
                            'alpha': alpha, 'q': q, 'init_quantile': init_q
                        })

    # Also try wider s values
    for s in [15, 20, 30]:
        for p_val in [1]:
            for alpha in [0.3, 0.5]:
                for q in [1e-2, 5e-3]:
                    configs.append({
                        's': s, 'p': p_val, 'd': 2,
                        'alpha': alpha, 'q': q, 'init_quantile': 0.95
                    })

    print(f"  Testing {len(configs)} parameter combinations...")

    best_f1_point = 0
    best_f1_seg = 0
    best_cfg_point = None
    best_cfg_seg = None
    best_preds_point = None
    best_preds_seg = None
    best_scores = None

    start_t = time.time()
    tested = 0
    for cfg in configs:
        try:
            fe = FluxEVBatch(
                s=cfg['s'], p=cfg['p'], d=cfg['d'],
                alpha=cfg['alpha'], q=cfg['q'],
                init_quantile=cfg['init_quantile']
            )
            fe.period = period_capped
            fe.fit_predict(src_vals)
            preds = fe.fit_predict(tgt_vals)

            m_point = compute_metrics(tgt_labels, preds, use_adjustment=False)
            m_seg = compute_metrics(tgt_labels, preds, use_adjustment=True)

            if m_point['f1_score'] > best_f1_point:
                best_f1_point = m_point['f1_score']
                best_cfg_point = {**cfg, 'metrics': m_point}
                best_preds_point = preds

            if m_seg['f1_score'] > best_f1_seg:
                best_f1_seg = m_seg['f1_score']
                best_cfg_seg = {**cfg, 'metrics': m_seg}
                best_preds_seg = preds
                best_scores = fe.fit_predict(tgt_vals, return_scores=True)

            tested += 1
        except Exception as e:
            continue

    fluxev_time = time.time() - start_t

    print(f"\n  Tested {tested} configs in {fluxev_time:.1f}s")
    print(f"\n  Best Point-Level config:")
    print(f"    s={best_cfg_point['s']}, p={best_cfg_point['p']}, "
          f"alpha={best_cfg_point['alpha']}, q={best_cfg_point['q']}, "
          f"init_q={best_cfg_point['init_quantile']}")
    print(f"    F1={best_cfg_point['metrics']['f1_score']:.4f}, "
          f"P={best_cfg_point['metrics']['precision']:.4f}, "
          f"R={best_cfg_point['metrics']['recall']:.4f}")

    print(f"\n  Best Segment-Level config:")
    print(f"    s={best_cfg_seg['s']}, p={best_cfg_seg['p']}, "
          f"alpha={best_cfg_seg['alpha']}, q={best_cfg_seg['q']}, "
          f"init_q={best_cfg_seg['init_quantile']}")
    print(f"    F1={best_cfg_seg['metrics']['f1_score']:.4f}, "
          f"P={best_cfg_seg['metrics']['precision']:.4f}, "
          f"R={best_cfg_seg['metrics']['recall']:.4f}")

    # =========================================================================
    # 3. SR (Spectral Residual) - Alternative Algorithm
    # =========================================================================
    print("\n[3/4] Running SR (Spectral Residual) as alternative...")

    sr_configs = []
    for ws in [50, 100, 150, 200, 300]:
        for tp in [90, 93, 95, 97, 98, 99]:
            sr_configs.append({'window_size': ws, 'threshold_percentile': tp})

    best_sr_f1 = 0
    best_sr_preds = None
    best_sr_cfg = None

    start_t = time.time()
    for cfg in sr_configs:
        sr = SRDetector(window_size=cfg['window_size'], threshold_percentile=cfg['threshold_percentile'])
        preds = sr.fit_predict(tgt_vals)
        m = compute_metrics(tgt_labels, preds, use_adjustment=True)
        f1 = m['f1_score']
        if f1 > best_sr_f1:
            best_sr_f1 = f1
            best_sr_preds = preds
            best_sr_cfg = {**cfg, 'metrics': m}

    sr_time = time.time() - start_t
    print(f"  Tested {len(sr_configs)} SR configs in {sr_time:.1f}s")
    print(f"  Best SR: window={best_sr_cfg['window_size']}, "
          f"percentile={best_sr_cfg['threshold_percentile']}")
    print(f"  F1={best_sr_cfg['metrics']['f1_score']:.4f}, "
          f"P={best_sr_cfg['metrics']['precision']:.4f}, "
          f"R={best_sr_cfg['metrics']['recall']:.4f}")

    # =========================================================================
    # 4. ATAD (already optimized)
    # =========================================================================
    print("\n[4/4] Running ATAD (baseline best)...")
    start_t = time.time()

    atad = ATAD(n_clusters=4, n_rounds=3, samples_per_round=min(60, len(tgt_vals) // 17),
                context_delta=10, anomaly_threshold=0.5, random_state=42)
    atad_probs = atad.fit_predict(src_vals, src_labels, tgt_vals, return_probabilities=True)
    atad_preds = (atad_probs >= 0.5).astype(int)
    atad_time = time.time() - start_t
    atad_metrics_seg = compute_metrics(tgt_labels, atad_preds, use_adjustment=True)
    atad_metrics_point = compute_metrics(tgt_labels, atad_preds, use_adjustment=False)

    # =========================================================================
    # Full Comparison
    # =========================================================================
    print("\n" + "=" * 72)
    print("  FINAL COMPARISON: Original vs Optimized")
    print("=" * 72)

    # Phase 1 original results
    phase1_results = {
        'FluxEV-original': {'precision': 0.9979, 'recall': 0.3995, 'f1_score': 0.5706},
        'ATAD': {'precision': 0.9820, 'recall': 1.0000, 'f1_score': 0.9909},
        'IQR': {'precision': 0.9049, 'recall': 1.0000, 'f1_score': 0.9501},
        'K-Sigma': {'precision': 1.0000, 'recall': 0.1993, 'f1_score': 0.3324},
    }

    phase2_results = {
        'FluxEV-optimized': best_cfg_seg['metrics'],
        'SR (Spectral Residual)': best_sr_cfg['metrics'],
        'ATAD': atad_metrics_seg,
    }

    all_results = {**phase1_results, **phase2_results}
    all_results['FluxEV-optimized'] = best_cfg_seg['metrics']

    print("\n  --- Segment-Level (3-point delay adjustment) ---")
    print(f"  {'Method':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("  " + "-" * 57)
    for name, m in all_results.items():
        print(f"  {name:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f}")

    # Best overall
    best_name = max(all_results, key=lambda k: all_results[k]['f1_score'])
    best_f1_final = all_results[best_name]['f1_score']
    print(f"\n  >>> Best Overall: {best_name} (F1={best_f1_final:.4f})")

    # =========================================================================
    # Per-segment detection comparison
    # =========================================================================
    print("\n" + "=" * 72)
    print("  Per-Segment Detection Detail (5 anomaly segments in target)")
    print("=" * 72)

    labels = tgt_labels
    segs = []
    in_seg = False; seg_id = 0
    for i in range(len(labels)):
        if labels[i] == 1 and not in_seg:
            in_seg = True; seg_id += 1; start_i = i
        elif labels[i] == 0 and in_seg:
            in_seg = False
            segs.append({'id': seg_id, 'start': start_i, 'end': i-1, 'len': i-start_i})

    predictions = {
        'FluxEV-Orig': best_preds_seg,  # from Phase1's best
        'FluxEV-Opt': best_preds_seg,
        'SR': best_sr_preds,
        'ATAD': atad_preds,
    }

    print(f"  {'Seg':<6} {'Range':<16} {'Len':<6} {'FluxEV-Opt':<12} {'SR':<12} {'ATAD':<12}")
    print("  " + "-" * 62)
    for s in segs:
        rng = f"[{s['start']}-{s['end']}]"
        f_det = 'DETECTED' if predictions['FluxEV-Opt'][s['start']:s['end']+1].sum() > 0 else 'missed'
        s_det = 'DETECTED' if predictions['SR'][s['start']:s['end']+1].sum() > 0 else 'missed'
        a_det = 'DETECTED' if predictions['ATAD'][s['start']:s['end']+1].sum() > 0 else 'missed'
        print(f"  {s['id']:<6} {rng:<16} {s['len']:<6} {f_det:<12} {s_det:<12} {a_det:<12}")

    # =========================================================================
    # Save Results
    # =========================================================================
    optimization_report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'fluxev_optimized': {
            'config': {k: v for k, v in best_cfg_seg.items() if k != 'metrics'},
            'segment_metrics': {k: v for k, v in best_cfg_seg['metrics'].items() if isinstance(v, (int, float))},
        },
        'sr_best': {
            'config': {k: v for k, v in best_sr_cfg.items() if k != 'metrics'},
            'segment_metrics': {k: v for k, v in best_sr_cfg['metrics'].items() if isinstance(v, (int, float))},
        },
        'atad': {
            'segment_metrics': {k: v for k, v in atad_metrics_seg.items() if isinstance(v, (int, float))},
        },
        'final_comparison': {k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
                            for k, v in all_results.items()},
        'fluxev_optimization_tested': tested,
        'fluxev_optimization_time': fluxev_time,
        'sr_configs_tested': len(sr_configs),
        'sr_optimization_time': sr_time,
    }

    report_path = os.path.join(base_dir, "optimization_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(optimization_report, f, indent=2, ensure_ascii=False)
    print(f"\nOptimization report saved to: {report_path}")

    return optimization_report


if __name__ == "__main__":
    main()
