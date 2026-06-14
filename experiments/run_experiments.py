"""SockShop — USAD + PatternMatcher 实验"""
import numpy as np, json, os, sys
sys.path.insert(0,os.path.dirname(__file__))
from usad import run_usad_experiment
from pattern_matcher import run_pattern_matcher

RD="F:/CCode/Homework/experiments/results"; os.makedirs(RD,exist_ok=True)

if __name__=="__main__":
    print("="*60)
    print("  SockShop 异常检测实验 (USAD + PatternMatcher)")
    print("="*60)

    # 生成 SockShop 模拟监控数据
    np.random.seed(42)
    nm, nt, nrc, inc_t = 30, 500, 3, 400
    svcs = ['front-end','carts','catalogue','orders','payment','user','shipping',
            'queue-master','rabbitmq','carts-db','orders-db','user-db','catalogue-db','edge-router']
    mn = [f"{s}.cpu" for s in svcs] + [f"{s}.memory" for s in svcs] + ["http.latency","error.rate"]
    nm = min(nm, len(mn))
    data = np.zeros((nm, nt))
    for i in range(nm): data[i] = np.random.uniform(20,60,nt) + np.random.randn(nt)*3
    rcs = np.random.choice(nm, nrc, replace=False)
    for idx in rcs:
        p = np.random.choice(['s_increase','s_decrease','steady'])
        d = np.random.randint(30,80); mag = np.random.uniform(80,150)
        if p == 's_increase': data[idx, inc_t:inc_t+d] += mag
        elif p == 's_decrease': data[idx, inc_t:inc_t+d] -= mag
        else: data[idx, inc_t-d:inc_t] += np.linspace(0,mag,d)
    labels = np.zeros(nt); labels[inc_t:] = 1
    metrics_data = data.T

    print(f"Data: {metrics_data.shape[1]} metrics x {metrics_data.shape[0]} points")
    print(f"Root causes: {nrc} @ t={inc_t}")

    # USAD
    sp = inc_t - 100
    train, test = metrics_data[:sp], metrics_data[sp:]
    tlb = labels[sp:]
    print(f"\n[USAD]")
    ur = run_usad_experiment(train, test, tlb, ws=10, ls=30, ep=80, ds=5)

    # PatternMatcher
    print(f"\n[PatternMatcher]")
    pr = run_pattern_matcher(metrics_data.T, inc_t, rcs)

    # Save
    report = {'dataset':{'n_metrics':nm,'n_timestamps':nt,'n_root_cause':nrc,'incident':inc_t},
              'usad': {'metrics': ur.get('metrics',{})},
              'pattern_matcher': pr.get('evaluation',{})}
    with open(f"{RD}/final_report.json","w") as f: json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {RD}/final_report.json")
