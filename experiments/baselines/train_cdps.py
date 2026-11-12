"""Portable entry point for the retained CDPS GPU implementation."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cdps_gpu import CDPS_GPU
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--dimension',type=int,default=128);p.add_argument('--fit-samples',type=int,default=2000);p.add_argument('--device',default='cuda');p.add_argument('--objective',default='native');p.add_argument('--iterations',type=int,default=300)
a=p.parse_args();data=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
if a.objective=='warpdistill':
    import cdps_gpu
    class RelativeLoss:
        def __getattr__(self,name):return getattr(F,name)
        def mse_loss(self,pred,target):
            keep=target>1e-6
            return F.huber_loss((pred[keep]-target[keep])/target[keep],torch.zeros_like(target[keep]),delta=.1)
    cdps_gpu.F=RelativeLoss()
model=CDPS_GPU(n_shapelets=a.dimension,shapelet_length=.2,max_iter=a.iterations,device=a.device)
start=time.perf_counter();model.fit(np.load(data/'train.npy')[:a.fit_samples]);fit=time.perf_counter()-start
# This legacy estimator stores tensors outside state_dict; use its own object
# serialization for local checkpoints and do not load untrusted model files.
torch.save(model,out/'model.pt')
for split,short in [('train','train'),('valid','valid'),('test','test'),('database','db'),('queries','q')]:
    np.save(out/f'emb_{short}.npy',model.transform(np.load(data/f'{split}.npy')))
(out/'fit.json').write_text(json.dumps({'seconds':fit,'dimension':a.dimension,'fit_samples':a.fit_samples,'objective':a.objective},indent=2)+'\n')
