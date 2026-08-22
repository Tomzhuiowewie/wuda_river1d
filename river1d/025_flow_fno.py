#!/usr/bin/env python3
"""标准全场监督二维 FNO：由初边值、河道几何和 Manning 场预测 Z/Q 全场。

本文件包含完整的模型、数据处理、训练、验证、测试和诊断流程，不依赖临时
实验脚本。训练目标只使用完整 Z/Q 场监督，PDE 残差仅在训练结束后诊断。
"""
from __future__ import annotations
import argparse, csv, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from river1d.config import CONFIG
from river1d._geometry import CrossSectionGeometry

CACHE = ROOT / "training_dataset_cache"
OUT = ROOT / "outputs/experiments/025_standard_fno"

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, mt, mx):
        super().__init__(); self.cin=cin; self.cout=cout; self.mt=mt; self.mx=mx
        scale=1.0/(cin*cout)
        self.w1r=nn.Parameter(scale*torch.randn(cin,cout,mt,mx)); self.w1i=nn.Parameter(scale*torch.randn(cin,cout,mt,mx))
        self.w2r=nn.Parameter(scale*torch.randn(cin,cout,mt,mx)); self.w2i=nn.Parameter(scale*torch.randn(cin,cout,mt,mx))
    def compl_mul(self, x, w): return torch.einsum("bixy,ioxy->boxy", x, w)
    def forward(self, x):
        # CUDA AMP does not support complex-half FFT/einsum reliably.  Keep
        # the spectral branch in float32 while the pointwise branches may use
        # autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x=x.float(); b,_,nt,nx=x.shape; ft=torch.fft.rfft2(x, norm="ortho")
            out=torch.zeros(b,self.cout,nt,nx//2+1, dtype=torch.cfloat, device=x.device)
            mt=min(self.mt,nt); mx=min(self.mx,nx//2+1)
            w1=torch.complex(self.w1r[:,:,:mt,:mx],self.w1i[:,:,:mt,:mx]); w2=torch.complex(self.w2r[:,:,:mt,:mx],self.w2i[:,:,:mt,:mx])
            out[:,:,:mt,:mx]=self.compl_mul(ft[:,:,:mt,:mx],w1)
            out[:,:,-mt:,:mx]=out[:,:,-mt:,:mx]+self.compl_mul(ft[:,:,-mt:,:mx],w2)
            return torch.fft.irfft2(out, s=(nt,nx), norm="ortho")

class FNOBlock(nn.Module):
    def __init__(self,w,mt,mx):
        super().__init__(); self.spec=SpectralConv2d(w,w,mt,mx); self.local=nn.Conv2d(w,w,1); self.norm=nn.GroupNorm(8,w)
    def forward(self,x): return F.gelu(self.norm(self.spec(x)+self.local(x))+x)

class FNO2d(nn.Module):
    def __init__(self, cin=9, width=64, mt=24, mx=16, layers=4):
        super().__init__(); self.lift=nn.Conv2d(cin,width,1); self.blocks=nn.ModuleList([FNOBlock(width,mt,mx) for _ in range(layers)]); self.proj=nn.Sequential(nn.Conv2d(width,128,1),nn.GELU(),nn.Conv2d(128,2,1))
    def forward(self,x):
        x=self.lift(x)
        for b in self.blocks: x=b(x)
        return self.proj(x)

def files():
    fs=sorted(CACHE.glob("S*_hydrodynamics_all_sections_15min_warmup3.npz"))
    return [p for i,p in enumerate(fs,1) if i%5], [p for i,p in enumerate(fs,1) if not i%5]

def raw(path):
    d=np.load(path); x=(d["stations"][0]-d["stations"])*1000.0; t=d["times"]*86400.0
    return x.astype(np.float32),t.astype(np.float32),d["z_grid"].astype(np.float32),d["q_grid"].astype(np.float32)

def grid_from_ref(path, nt=256, nx=128):
    x,t,z,q=raw(path); xu=np.linspace(float(x.min()),float(x.max()),nx,dtype=np.float32); ti=np.linspace(0,len(t)-1,nt).round().astype(int); tu=t[ti]
    return x,t,z,q,xu,tu,ti

def interp_field(x,t,a,xu,ti):
    # time interpolation followed by spatial interpolation onto the regular FNO grid
    at=a[ti]; tmp=np.stack([np.interp(xu,x,row) for row in at],axis=0)
    return tmp.astype(np.float32)

def sample(path,xu,ti,geom,stats):
    x,t,z,q=raw(path); zn=interp_field(x,t,z,xu,ti); qn=interp_field(x,t,q,xu,ti)
    # Inputs are allowed conditions/static fields only; target fields are returned separately.
    q_up=qn[:,0]; z_dn=zn[:,-1]; z0=zn[0]; q0=qn[0]
    xx=np.broadcast_to((2*(xu-xu.min())/(xu.max()-xu.min())-1)[None,:],(len(ti),len(xu)))
    tt=np.broadcast_to((2*(np.linspace(t[0],t[-1],len(ti))-t[0])/(t[-1]-t[0])-1)[:,None],(len(ti),len(xu)))
    bed=stats['bed']; upper=stats['upper']; man=np.full_like(xx,0.016)
    chans=np.stack([xx,tt,
        np.broadcast_to(((q_up-stats['qmean'])/stats['qstd'])[:,None],xx.shape),
        np.broadcast_to(((z_dn-stats['zmean'])/stats['zstd'])[:,None],xx.shape),
        np.broadcast_to(((z0-stats['zmean'])/stats['zstd'])[None,:],xx.shape),
        np.broadcast_to(((q0-stats['qmean'])/stats['qstd'])[None,:],xx.shape),
        np.broadcast_to(((bed-stats['zmean'])/stats['zstd'])[None,:],xx.shape),
        np.broadcast_to(((upper-stats['zmean'])/stats['zstd'])[None,:],xx.shape),man],axis=0)
    y=np.stack([(zn-stats['zmean'])/stats['zstd'],(qn-stats['qmean'])/stats['qstd']],axis=0)
    return torch.from_numpy(chans),torch.from_numpy(y),zn,qn

def fit_stats(train,ref,xu,ti,geom):
    x,t,z,q=raw(ref)
    with torch.no_grad(): bed,upper=geom.stage_bounds(torch.from_numpy(xu))
    zs=[];qs=[]
    for p in train:
        _,_,zz,qq=raw(p); zs.append(interp_field(x,t,zz,xu,ti)); qs.append(interp_field(x,t,qq,xu,ti))
    zz=np.concatenate(zs); qq=np.concatenate(qs)
    return {'zmean':float(zz.mean()),'zstd':float(max(zz.std(),1e-6)),'qmean':float(qq.mean()),'qstd':float(max(qq.std(),1e-6)),'bed':bed.numpy(),'upper':upper.numpy()}

def batches(paths,batch):
    for i in range(0,len(paths),batch): yield paths[i:i+batch]

@torch.no_grad()
def evaluate(model,paths,xu,ti,stats,device):
    pz=[];pq=[];tz=[];tq=[]
    for group in batches(paths,2):
        xs=[];ys=[];rawz=[];rawq=[]
        for p in group:
            a,b,z,q=sample(p,xu,ti, None,stats); xs.append(a);ys.append(b);rawz.append(z);rawq.append(q)
        pred=model(torch.stack(xs).to(device)).cpu().numpy(); y=np.stack(ys)
        pz.append(pred[:,0]*stats['zstd']+stats['zmean']); pq.append(pred[:,1]*stats['qstd']+stats['qmean']); tz.append(np.stack(rawz)); tq.append(np.stack(rawq))
    pz,pq,tz,tq=map(lambda v:np.concatenate(v),[pz,pq,tz,tq])
    def l2(a,b): return float(100*np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-8))
    return {'z_l2':l2(pz,tz),'q_l2':l2(pq,tq),'z_nse':float(1-np.sum((pz-tz)**2)/max(np.sum((tz-tz.mean())**2),1e-8)),'q_nse':float(1-np.sum((pq-tq)**2)/max(np.sum((tq-tq.mean())**2),1e-8))}

@torch.no_grad()
def regional_evaluate(model,paths,xu,ti,stats,device):
    vals={k:[[],[],[],[]] for k in ('all','upstream','downstream','x46')}
    for p in paths:
        x,t,z,q=raw(p); a,_,zz,qq=sample(p,xu,ti,None,stats); pred=model(a[None].to(device)).cpu().numpy()[0]; pz=pred[0]*stats['zstd']+stats['zmean']; pq=pred[1]*stats['qstd']+stats['qmean']; ix=int(np.argmin(abs(xu-x[46])))
        true46z=np.asarray([np.interp(x[46],x,row) for row in z[ti]]); true46q=np.asarray([np.interp(x[46],x,row) for row in q[ti]])
        items={'all':(pz,pq,zz,qq),'upstream':(pz[:,0],pq[:,0],zz[:,0],qq[:,0]),'downstream':(pz[:,-1],pq[:,-1],zz[:,-1],qq[:,-1]),'x46':(pz[:,ix],pq[:,ix],true46z,true46q)}
        for k,(a1,a2,b1,b2) in items.items(): vals[k][0].append(a1); vals[k][1].append(a2); vals[k][2].append(b1); vals[k][3].append(b2)
    out={}
    for k,(a1,a2,b1,b2) in vals.items():
        a1,a2,b1,b2=map(lambda v:np.concatenate([x.reshape(-1) for x in v]),(a1,a2,b1,b2)); out[k]={'z_l2':float(100*np.linalg.norm(a1-b1)/np.linalg.norm(b1)),'q_l2':float(100*np.linalg.norm(a2-b2)/np.linalg.norm(b2))}
    return out

@torch.no_grad()
def pde_diagnostics(model,paths,xu,ti,stats,device,geom):
    xg=torch.from_numpy(xu); dx=float(xu[1]-xu[0]); tref=raw(paths[0])[1]; dt=float(np.median(np.diff(np.linspace(tref[0],tref[-1],len(ti))))); vals={'mass':[],'momentum':[]}
    for p in paths:
        a,_,_,_=sample(p,xu,ti,None,stats); pred=model(a[None].to(device)).cpu().numpy()[0]; z=pred[0]*stats['zstd']+stats['zmean']; q=pred[1]*stats['qstd']+stats['qmean']; area,_,per=geom(xg,torch.from_numpy(z)); area=area.numpy(); per=per.numpy(); mass=np.gradient(area,dt,axis=0)+np.gradient(q,dx,axis=1); flux=q*q/np.maximum(area,1e-6); mom=np.gradient(q,dt,axis=0)+np.gradient(flux,dx,axis=1)+9.81*area*(np.gradient(z,dx,axis=1)+0.016**2*q*np.abs(q)/(np.maximum(area,1e-6)**2*np.maximum(area/per,1e-6)**(4/3))); vals['mass'].append(abs(mass).ravel()); vals['momentum'].append(abs(mom).ravel())
    out={}
    for k,v in vals.items():
        a=np.concatenate(v); out[k]={'mse':float(np.mean(a*a)),'mean':float(np.mean(a)),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'p95':float(np.percentile(a,95)),'max':float(np.max(a))}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=40); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--resume',action='store_true'); ap.add_argument('--seed',type=int,default=2032); ap.add_argument('--out',default=str(OUT)); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; formal FNO run is intentionally stopped.')
    seed_all(args.seed); device=torch.device('cuda'); train,test=files(); rng=np.random.default_rng(args.seed); order=rng.permutation(len(train)); val=[train[i] for i in order[:40]]; tr=[train[i] for i in order[40:]]
    ref=train[0]; x,t,z,q=raw(ref); nx,nt=128,256; xu=np.linspace(x.min(),x.max(),nx,dtype=np.float32); ti=np.linspace(0,len(t)-1,nt).round().astype(int)
    geom=CrossSectionGeometry(CONFIG.cross_section_path,torch.from_numpy(x),device='cpu'); stats=fit_stats(tr,ref,xu,ti,geom)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); (out/'config.json').write_text(json.dumps({'seed':args.seed,'train':len(tr),'validation':len(val),'test':len(test),'grid':[nt,nx],'architecture':{'width':64,'modes_t':24,'modes_x':16,'layers':4},'formal_gpu':True},indent=2))
    model=FNO2d().to(device); lr0,lrmin=1e-3,1e-5; opt=torch.optim.AdamW(model.parameters(),lr=lr0/5,weight_decay=1e-4); scaler=torch.amp.GradScaler('cuda',enabled=True); best=1e99; history=[]; epochs=1 if args.smoke else args.epochs; start_epoch=1
    if args.resume and (Path(args.out)/'best_model.pt').exists():
        model.load_state_dict(torch.load(Path(args.out)/'best_model.pt',map_location=device,weights_only=True)); start_epoch=41
        old=Path(args.out)/'history.json'
        if old.exists(): history=json.loads(old.read_text())
        best=min((max(h['z_l2'],h['q_l2']) for h in history),default=1e99)
    for ep in range(start_epoch,epochs+1):
        if args.resume: lr=lrmin
        elif ep<=5: lr=lr0*ep/5
        else: lr=lrmin+0.5*(lr0-lrmin)*(1+math.cos(math.pi*(ep-5)/max(1,epochs-5)))
        for group in opt.param_groups: group['lr']=lr
        model.train(); losses=[]; start=time.time(); rng.shuffle(tr)
        for group in batches(tr[:8] if args.smoke else tr,2):
            xx=[];yy=[]
            for p in group:
                a,b,_,_=sample(p,xu,ti,None,stats);xx.append(a);yy.append(b)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16): loss=F.mse_loss(model(torch.stack(xx).to(device)),torch.stack(yy).to(device))
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        model.eval(); v=evaluate(model,val if not args.smoke else val[:2],xu,ti,stats,device); score=max(v['z_l2'],v['q_l2']); history.append({'epoch':ep,'loss':float(np.mean(losses)),'learning_rate':lr,'seconds':time.time()-start,**v}); print(json.dumps(history[-1]),flush=True)
        if score<best: best=score; torch.save(model.state_dict(),out/'best_model.pt')
    (out/'history.json').write_text(json.dumps(history,indent=2));
    with (out/'history.csv').open('w',newline='') as f:
        keys=sorted({k for h in history for k in h}); w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(history)
    model.load_state_dict(torch.load(out/'best_model.pt',map_location=device,weights_only=True)); model.eval(); result={'validation':evaluate(model,val,xu,ti,stats,device),'test':evaluate(model,test,xu,ti,stats,device),'parameters':sum(p.numel() for p in model.parameters()),'best_validation_score':best}; (out/'result.json').write_text(json.dumps(result,indent=2));
    (out/'regional_metrics.json').write_text(json.dumps({'validation':regional_evaluate(model,val,xu,ti,stats,device),'test':regional_evaluate(model,test,xu,ti,stats,device)},indent=2)); (out/'pde_diagnostics.json').write_text(json.dumps({'test':pde_diagnostics(model,test,xu,ti,stats,device,geom),'note':'post-hoc finite-difference diagnostic; excluded from training loss'},indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
