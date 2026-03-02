from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
import torch.nn as nn

# Streamlit demo for flow matching on a noisy Gaussian mixture model.

@dataclass
class DataCfg:
    k:int; n:int; noise_std:float; seed:int
    layout:str="Ring (structured)"
    r:float=2.3; comp_std:float=0.25; skew:float=0.8
    mean_span:float=3.0; std_min:float=0.10; std_max:float=0.50; weight_alpha:float=0.7

@dataclass
class TrainCfg:
    source:str; hidden:int; layers:int; lr:float; steps:int; batch:int; integ:int

class VelNet(nn.Module):
    def __init__(self, h:int, l:int):
        """Small MLP that predicts 2D velocity given position and time."""
        super().__init__()
        seq=[nn.Linear(3,h),nn.SiLU()]
        for _ in range(l-1): seq += [nn.Linear(h,h),nn.SiLU()]
        seq += [nn.Linear(h,2)]
        self.net=nn.Sequential(*seq)
    def forward(self,x,t): return self.net(torch.cat([x,t],dim=-1))

def make_gmm(c:DataCfg):
    """Create clean/noisy samples from a 2D GMM (ring or randomized components)."""
    rng=np.random.default_rng(c.seed)
    if c.layout=="Randomized components":
        m=rng.uniform(-c.mean_span,c.mean_span,size=(c.k,2))
        s_min,s_max=sorted((c.std_min,c.std_max)); comp_stds=rng.uniform(s_min,max(s_max,s_min+1e-6),size=c.k)
        w=rng.dirichlet(np.full(c.k,max(c.weight_alpha,1e-3)))
    else:
        a=np.linspace(0,2*np.pi,c.k,endpoint=False)+rng.uniform(0,2*np.pi)
        m=np.c_[c.r*np.cos(a),c.r*np.sin(a)] + rng.normal(0,0.15,(c.k,2))
        logits=rng.normal(size=c.k)*c.skew; w=np.exp(logits-logits.max()); w/=w.sum()
        comp_stds=np.full(c.k,c.comp_std)
    comp=rng.choice(c.k,size=c.n,p=w)
    clean=m[comp]+rng.normal(0,comp_stds[comp][:,None],(c.n,2)); noisy=clean+rng.normal(0,c.noise_std,clean.shape)
    return clean.astype(np.float32), noisy.astype(np.float32), m, w

def sw2d(x,y,nproj=64,seed=0):
    """Approximate sliced Wasserstein distance between two 2D sample sets."""
    rng=np.random.default_rng(seed); d=rng.normal(size=(nproj,2)); d/=np.linalg.norm(d,axis=1,keepdims=True)+1e-8
    return float(np.mean([np.mean(np.abs(np.sort(x@u)-np.sort(y@u))) for u in d]))

def train_flow(clean_np,noisy_np,c:TrainCfg,seed:int):
    """Train velocity field with flow matching and integrate it to generate outputs."""
    torch.manual_seed(seed)
    clean=torch.tensor(clean_np,dtype=torch.float32); noisy=torch.tensor(noisy_np,dtype=torch.float32)
    net=VelNet(c.hidden,c.layers); opt=torch.optim.Adam(net.parameters(),lr=c.lr); losses=[]
    for _ in range(c.steps):
        idx=torch.randint(0,len(clean),(c.batch,)); x1=clean[idx]
        x0=noisy[idx] if c.source=="Noisy observation -> Clean target" else torch.randn_like(x1)
        t=torch.rand(c.batch,1); xt=(1-t)*x0+t*x1; v=x1-x0
        loss=((net(xt,t)-v)**2).mean(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.item()))
    with torch.no_grad():
        x=torch.tensor(noisy_np,dtype=torch.float32) if c.source=="Noisy observation -> Clean target" else torch.randn((len(clean_np),2))
        start=x.clone().numpy(); dt=1.0/c.integ
        for k in range(c.integ):
            t=torch.full((len(x),1),k/c.integ); x=x+net(x,t)*dt
    return losses,start,x.numpy()

def fig_data(clean,noisy,means):
    """Visualize clean targets and noisy observations side-by-side."""
    f=make_subplots(rows=1,cols=2,subplot_titles=("Clean target","Noisy observation"))
    f.add_trace(go.Scattergl(x=clean[:,0],y=clean[:,1],mode="markers",marker=dict(size=4,opacity=0.5)),row=1,col=1)
    f.add_trace(go.Scatter(x=means[:,0],y=means[:,1],mode="markers",marker=dict(size=9,symbol="x")),row=1,col=1)
    f.add_trace(go.Scattergl(x=noisy[:,0],y=noisy[:,1],mode="markers",marker=dict(size=4,opacity=0.5,color="#FF6B6B")),row=1,col=2)
    f.update_layout(height=360,showlegend=False); f.update_xaxes(scaleanchor="y",scaleratio=1); return f

def fig_res(start,gen,clean):
    """Visualize source samples, generated samples, and their overlay with targets."""
    f=make_subplots(rows=1,cols=3,subplot_titles=("Source t=0","Generated t=1","Overlay"))
    f.add_trace(go.Scattergl(x=start[:,0],y=start[:,1],mode="markers",marker=dict(size=4,opacity=0.45)),row=1,col=1)
    f.add_trace(go.Scattergl(x=gen[:,0],y=gen[:,1],mode="markers",marker=dict(size=4,opacity=0.45,color="#6A5ACD")),row=1,col=2)
    f.add_trace(go.Scattergl(x=clean[:,0],y=clean[:,1],mode="markers",marker=dict(size=3,opacity=0.2,color="#4CAF50")),row=1,col=3)
    f.add_trace(go.Scattergl(x=gen[:,0],y=gen[:,1],mode="markers",marker=dict(size=4,opacity=0.45,color="#6A5ACD")),row=1,col=3)
    f.update_layout(height=390,showlegend=False); f.update_xaxes(scaleanchor="y",scaleratio=1); return f

def main():
    """Build Streamlit UI, run training, and display diagnostics for the demo."""
    st.set_page_config(page_title="Flow Matching GMM Demo",layout="wide")
    st.title("Flow Matching Demo: Noisy Gaussian Mixture")
    with st.sidebar:
        seed=st.number_input("Seed",0,999999,42,1); k=st.slider("# Components",2,10,4); n=st.slider("# Samples",300,6000,2000,100)
        layout=st.radio("GMM layout",["Ring (structured)","Randomized components"])
        if layout=="Randomized components":
            mean_span=st.slider("Mean sampling span",0.5,6.0,3.0)
            std_min=st.slider("Component std (min)",0.02,1.0,0.10)
            std_max=st.slider("Component std (max)",0.03,1.5,0.50)
            weight_alpha=st.slider("Weight concentration (Dirichlet α)",0.05,5.0,0.70)
            r,comp_std,skew=2.3,0.25,0.8
        else:
            r=st.slider("Component separation",0.5,5.0,2.3); comp_std=st.slider("Target std",0.05,1.0,0.25)
            skew=st.slider("Mixture imbalance",0.0,2.5,0.8)
            mean_span,std_min,std_max,weight_alpha=3.0,0.10,0.50,0.70
        noise_std=st.slider("Observation noise std",0.0,2.0,0.7)
        src=st.radio("Source",["Noisy observation -> Clean target","Standard Gaussian -> Clean target"])
        hidden=st.select_slider("Hidden width",options=[32,64,96,128,192],value=96); layers=st.slider("Layers",1,5,2)
        lr=st.select_slider("LR",options=[1e-4,2e-4,5e-4,1e-3,2e-3],value=1e-3); steps=st.slider("Train steps",100,4000,1200,100)
        batch=st.select_slider("Batch",options=[64,128,256,512,1024],value=256); integ=st.slider("Integration steps",10,200,80)
        run=st.button("Train / Retrain",type="primary")
    clean,noisy,means,w=make_gmm(DataCfg(k,n,noise_std,int(seed),layout,r,comp_std,skew,mean_span,std_min,std_max,weight_alpha))
    st.plotly_chart(fig_data(clean,noisy,means),use_container_width=True)
    a,b,c=st.columns(3); a.metric("Noise",f"{noise_std:.2f}"); b.metric("Largest weight",f"{w.max():.2f}")
    c.metric("Target std",f"{comp_std:.2f}" if layout=="Ring (structured)" else f"{std_min:.2f}–{std_max:.2f}")
    if run:
        losses,start,gen=train_flow(clean,noisy,TrainCfg(src,hidden,layers,lr,steps,batch,integ),int(seed))
        st.session_state["res"]=(losses,start,gen,clean,sw2d(gen,clean,64,int(seed)+7))
    if "res" in st.session_state:
        losses,start,gen,clean_ref,sw=st.session_state["res"]
        st.metric("Sliced Wasserstein (lower is better)",f"{sw:.4f}")
        st.plotly_chart(fig_res(start,gen,clean_ref),use_container_width=True)
        lf=go.Figure(); lf.add_trace(go.Scatter(y=losses,mode="lines")); lf.update_layout(height=220,xaxis_title="Step",yaxis_title="Velocity MSE")
        st.plotly_chart(lf,use_container_width=True)
    st.markdown("- Increase **observation noise std**: harder denoising.\n- Reduce **width/layers**: underfitting.\n- Reduce **train steps**: poorer convergence.\n- Use **Standard Gaussian source**: generation from pure noise.")

if __name__ == "__main__":
    main()
