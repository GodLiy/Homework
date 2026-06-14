"""KDD20-USAD: UnSupervised Anomaly Detection on Multivariate Time Series"""
import numpy as np, torch, torch.nn as nn, torch.optim as optim, json, time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size//2), nn.ReLU(),
            nn.Linear(input_size//2, input_size//4), nn.ReLU(),
            nn.Linear(input_size//4, latent_size), nn.ReLU())
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, input_size//4), nn.ReLU(),
            nn.Linear(input_size//4, input_size//2), nn.ReLU(),
            nn.Linear(input_size//2, input_size), nn.Sigmoid())
    def forward(self, z): return self.net(z)

class USAD(nn.Module):
    def __init__(self, input_size, latent_size=40):
        super().__init__()
        self.encoder = Encoder(input_size, latent_size)
        self.decoder1 = Decoder(input_size, latent_size)
        self.decoder2 = Decoder(input_size, latent_size)
    def forward(self, w):
        z = self.encoder(w)
        return self.decoder1(z), self.decoder2(z), self.decoder2(self.encoder(self.decoder1(z)))
    def compute_anomaly_score(self, w, alpha=0.5, beta=0.5):
        self.eval()
        with torch.no_grad():
            w1, _, w2p = self.forward(w)
            return alpha*torch.mean((w-w1)**2, dim=1)+beta*torch.mean((w-w2p)**2, dim=1)

def train_usad(model, loader, epochs, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    hist = {'epoch':[],'loss1':[],'loss2':[]}
    model.train()
    for ep in range(1,epochs+1):
        l1,l2=0,0; n=ep
        for (bw,) in loader:
            bw=bw.float(); w1,w2,w2p=model(bw)
            c=1.0/n
            loss1=c*torch.mean((bw-w1)**2)+(1-c)*torch.mean((bw-w2p)**2)
            loss2=c*torch.mean((bw-w2)**2)-(1-c)*torch.mean((bw-w2p)**2)
            opt.zero_grad(); loss1.backward(retain_graph=True); loss2.backward(); opt.step()
            l1+=loss1.item(); l2+=loss2.item()
        hist['epoch'].append(ep); hist['loss1'].append(l1/len(loader)); hist['loss2'].append(l2/len(loader))
        if ep%10==0: print(f"  E{ep}/{epochs}: L1={hist['loss1'][-1]:.4f} L2={hist['loss2'][-1]:.4f}")
    return hist

def create_windows(data, ws):
    return np.array([data[i:i+ws].flatten() for i in range(len(data)-ws+1)])

def preprocess(data, ws=12, ds=5, scaler=None):
    n,d=data.shape; ds_data=np.zeros((n//ds,d))
    for i in range(n//ds):
        s=i*ds; e=min(s+ds,n); ds_data[i]=np.median(data[s:e],axis=0)
    if scaler is None: scaler=StandardScaler(); ds_data=scaler.fit_transform(ds_data)
    else: ds_data=scaler.transform(ds_data)
    return create_windows(ds_data,ws), scaler

def run_usad_experiment(train, test, labels=None, ws=12, ls=40, ep=80, ds=5, alpha=0.5, beta=0.5, th_p=95):
    print(f"USAD: dim={train.shape[1]} ws={ws} ls={ls} epochs={ep}")
    tw, sc = preprocess(train,ws,ds); ttw,_ = preprocess(test,ws,ds,sc)
    isz=tw.shape[1]; model=USAD(isz,ls)
    tl=torch.tensor(tw,dtype=torch.float32)
    hist=train_usad(model,DataLoader(TensorDataset(tl),64,True),ep)
    ttl=torch.tensor(ttw,dtype=torch.float32)
    trs=model.compute_anomaly_score(tl,alpha,beta).numpy()
    th=np.percentile(trs,th_p)
    scs=model.compute_anomaly_score(ttl,alpha,beta).numpy()
    preds=(scs>th).astype(int)
    # map back to points
    pp=np.zeros(len(test)); pc=np.zeros(len(test))
    for i in range(len(preds)):
        si=i*ds; ei=min(si+ws*ds,len(test))
        pp[si:ei]+=preds[i]; pc[si:ei]+=1
    pc[pc==0]=1; pp=(pp/pc>0.5).astype(int)
    r={'threshold':float(th),'anomaly_scores':scs.tolist(),'predictions':pp.tolist()}
    if labels is not None:
        ml=min(len(pp),len(labels)); p=pp[:ml]; t=labels[:ml]
        tp=np.sum((p==1)&(t==1)); fp=np.sum((p==1)&(t==0)); fn=np.sum((p==0)&(t==1))
        prec=tp/(tp+fp) if tp+fp>0 else 0; rec=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0
        r['metrics']={'precision':float(prec),'recall':float(rec),'f1':float(f1),'tp':int(tp),'fp':int(fp),'fn':int(fn)}
        print(f"  P={prec:.4f} R={rec:.4f} F1={f1:.4f} TP={tp} FP={fp}")
    return r

if __name__=="__main__": print("USAD ready")
