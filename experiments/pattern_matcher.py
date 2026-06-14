"""ISSRE21-PatternMatcher: Root-Cause Metric Identification"""
import numpy as np, json, time
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# Step 1: KS-test
def ks_test(metric, inc_t, l1=10, l2=30, alpha=0.05):
    sa=metric[max(0,inc_t-l1):inc_t]
    sn=metric[max(0,inc_t-l1-l2):max(0,inc_t-l1)]
    if len(sa)<3 or len(sn)<3: return False,1.0,0.0
    _,pv=stats.ks_2samp(sa,sn)
    return pv<alpha, pv, -np.log10(max(pv,0.0001))

def coarse_detection(data, inc_t, l1=10, l2=30, alpha=0.05):
    return [{'idx':i,'anom':r[0],'pval':r[1],'score':r[2]}
            for i in range(data.shape[0])
            for r in [ks_test(data[i],inc_t,l1,l2,alpha)]]

# Step 2: Pattern classification
class PatternClassifier:
    PATTERNS=['sudden_increase','sudden_decrease','level_shift_up','level_shift_down',
              'steady_increase','steady_decrease','single_spike','single_dip',
              'transient_ls_up','transient_ls_down','multiple_spikes','multiple_dips','fluctuations']
    WEIGHTS={p:0.8 for p in PATTERNS[:6]}; WEIGHTS.update({p:0.2 for p in PATTERNS[6:]})

    def __init__(self): self.scaler=MinMaxScaler()

    def classify(self, seg):
        if len(seg)<5: return 'fluctuations',0.2
        seg=np.array(seg).reshape(-1,1); seg=self.scaler.fit_transform(seg).flatten()
        n=len(seg); h1=seg[:n//2]; h2=seg[n//2:]
        trend=np.polyfit(np.arange(n),seg,1)[0]
        m1,m2=np.mean(h1),np.mean(h2); std=np.std(seg)
        from scipy.signal import find_peaks
        peaks,_=find_peaks(seg,distance=n//5); dips,_=find_peaks(-seg,distance=n//5)
        if abs(trend)>0.02:
            if trend>0:
                return ('level_shift_up',0.8) if m2>m1*1.5 else ('steady_increase',0.8)
            return ('level_shift_down',0.8) if m2<m1*0.5 else ('steady_decrease',0.8)
        if m2>m1*2.0: return 'sudden_increase',0.8
        if m2<m1*0.5: return 'sudden_decrease',0.8
        if len(peaks)==1: return 'single_spike',0.2
        if len(dips)==1: return 'single_dip',0.2
        if len(peaks)>=3: return 'multiple_spikes',0.2
        if len(dips)>=3: return 'multiple_dips',0.2
        if m2>m1*1.3: return 'transient_ls_up',0.2
        if m2<m1*0.7: return 'transient_ls_down',0.2
        return 'fluctuations',0.2

pc=PatternClassifier()

# Step 3: Ranking
def rank_metrics(anom_results, metrics, inc_t, w=30):
    rankings=[]
    for r in anom_results:
        if not r['anom']: continue
        seg=metrics[r['idx']][max(0,inc_t-w):inc_t]
        pn,pw=pc.classify(seg)
        rankings.append({'idx':r['idx'],'score':r['score']*pw,'anom_score':r['score'],
                        'pval':r['pval'],'pattern':pn,'weight':pw,
                        'type':'Type-1' if pw>0.5 else 'Type-2'})
    rankings.sort(key=lambda x:x['score'],reverse=True)
    return rankings

def run_pattern_matcher(metrics, inc_t, gt=None, l1=10, l2=30, w=30, alpha=0.05):
    print(f"PatternMatcher: {metrics.shape[0]} metrics x {metrics.shape[1]} points, t={inc_t}")
    t0=time.time()
    ar=coarse_detection(metrics, inc_t, l1, l2, alpha)
    na=sum(1 for r in ar if r['anom'])
    t1=time.time()
    ranking=rank_metrics(ar, metrics, inc_t, w)
    t2=time.time()
    r={'n_candidate':metrics.shape[0],'n_anom':na,'ranking':ranking,
       'timing':{'step1':t1-t0,'step2':t2-t1,'total':time.time()-t0}}
    if gt is not None and len(gt)>0:
        gs=set(gt); ac={}
        for k in [1,2,3,5]:
            tk=set(r['idx'] for r in ranking[:k]); ac[f'AC@{k}']=len(tk&gs)/k if k>0 else 0
        ac['Avg@3']=(ac['AC@1']+ac['AC@2']+ac['AC@3'])/3
        r['evaluation']=ac
        print(f"  AC@1={ac['AC@1']:.2f} Avg@3={ac['Avg@3']:.2f}")
    print(f"  Time: {r['timing']['total']:.3f}s")
    return r

def generate_synthetic_metrics(nm=50, nt=500, nrc=3, inc_t=400):
    np.random.seed(42); data=np.zeros((nm,nt))
    for i in range(nm): data[i]=np.random.uniform(20,60,nt)+np.random.randn(nt)*3
    rcs=np.random.choice(nm,nrc,replace=False)
    for idx in rcs:
        p=np.random.choice(['spike','shift','increase'])
        if p=='spike': data[idx,inc_t:inc_t+15]+=np.random.uniform(60,120,15)
        elif p=='shift': data[idx,inc_t:]+=np.random.uniform(50,100)
        else:
            d=np.random.randint(30,80)
            data[idx,inc_t-d:inc_t]+=np.linspace(0,np.random.uniform(50,120),d)
    return data, rcs

if __name__=="__main__": print("PatternMatcher ready")
