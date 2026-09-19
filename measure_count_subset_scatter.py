"""Fixed-background q_i(sum_j K_ij g_j) scatter versus latent count."""
import argparse
import csv
import json
from pathlib import Path
import socket
import subprocess

import numpy as np
import torch

from measure_context_transfer import locally_reachable_pairs
from src.config import ModelConfig
from src.context_transfer import encoder_parameters, evaluation_state, source_gradient, transfer
from src.data import enumerate_paths
from src.graph import Graph
from src.model import ToyTransformer


def balanced_slots(strata, n, seed):
    rng=np.random.default_rng(seed);values=np.unique(strata)
    quotas=np.full(len(values),n//len(values));quotas[:n%len(values)]+=1
    pieces=[]
    for value,quota in zip(values,quotas):
        pool=np.flatnonzero(strata==value)
        if len(pool)<quota:raise ValueError(f"insufficient parent {value}: {len(pool)} < {quota}")
        pieces.append(rng.choice(pool,int(quota),replace=False))
    slots=np.concatenate(pieces);rng.shuffle(slots);return slots


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--latent',type=int,default=78);p.add_argument('--batch-size',type=int,default=2048)
    p.add_argument('--counts',type=int,nargs='+',default=list(range(0,65,4)))
    p.add_argument('--subsets',type=int,default=128);p.add_argument('--eval-points',type=int,default=4)
    p.add_argument('--microbatch',type=int,default=256)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('submit to GPU compute node')
    if a.out_dir.exists():raise FileExistsError(a.out_dir)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    source=json.loads((a.run/'history.json').read_text());li=source['latents'].index(a.latent)
    graph=Graph.load(a.run/'graph.pt');paths=enumerate_paths(graph);banks=torch.load(a.run/'banks.pt',weights_only=True)
    pair_ids,parents=locally_reachable_pairs(paths,banks['train'],graph,source['config']['graph_layer'],a.latent,31001+a.latent)
    slots=balanced_slots(parents,a.batch_size,32001+a.latent);pairs=pair_ids[slots]
    off,on=torch.as_tensor(pairs[:,0]),torch.as_tensor(pairs[:,1])
    edges,labels=paths['edge_seqs'],paths['labels'];eval_ids=banks['test'][li,:a.eval_points,1]
    row=min(source['history'],key=lambda r:abs(r['gain_raw'][li]-.5));step=row['step']
    model=ToyTransformer(ModelConfig(**source['model_config'])).cuda().eval()
    model.load_state_dict(torch.load(a.run/'models'/f'step{step:06d}.pt',map_location='cpu',weights_only=True))
    params=encoder_parameters(model)
    eval_edges,eval_labels=edges[eval_ids].cuda(),labels[eval_ids].cuda()
    eval_directions=[source_gradient(model,params,eval_edges[i:i+1],eval_labels[i:i+1]) for i in range(a.eval_points)]
    all_ids=torch.cat([off,on]);scores=torch.empty(a.eval_points,len(all_ids),dtype=torch.float64)
    for start in range(0,len(all_ids),a.microbatch):
        ids=all_ids[start:start+a.microbatch];x,y=edges[ids].cuda(),labels[ids].cuda()
        _,g,_=evaluation_state(model,params,x,y)
        for ei,direction in enumerate(eval_directions):
            t=transfer(model,params,x,direction)
            scores[ei,start:start+len(ids)]=(g.double()*t.double()).flatten(1).sum(1).cpu()
    baseline=scores[:,:a.batch_size].sum(1);delta=scores[:,a.batch_size:]-scores[:,:a.batch_size]
    rng=np.random.default_rng(33001+a.latent);records=[]
    for count in a.counts:
        repeats=1 if count==0 else a.subsets
        for subset in range(repeats):
            chosen=np.empty(0,dtype=np.int64) if count==0 else rng.choice(a.batch_size,count,replace=False)
            extra=delta[:,torch.as_tensor(chosen)].sum(1) if count else torch.zeros(a.eval_points,dtype=torch.float64)
            total=baseline+extra
            for ei,eid in enumerate(eval_ids.tolist()):
                records.append(dict(seed=source['config']['seed'],step=step,gain=row['gain_raw'][li],
                    latent=a.latent,frequency=source['frequency'][li],eval_id=eid,count=count,subset=subset,
                    q_sum=float(total[ei]),q_extra=float(extra[ei]),q_baseline=float(baseline[ei]),
                    subset_examples=';'.join(map(str,chosen.tolist()))))
    a.out_dir.mkdir(parents=True)
    with (a.out_dir/'scatter.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,records[0]);w.writeheader();w.writerows(records)
    metadata=dict(complete=True,seed=source['config']['seed'],latent=a.latent,frequency=source['frequency'][li],
        natural_expected_count=source['frequency'][li]*a.batch_size,step=step,gain=row['gain_raw'][li],
        batch_size=a.batch_size,counts=a.counts,subsets=a.subsets,eval_ids=eval_ids.tolist(),
        pair_pool_size=len(pair_ids),eligible_parents=sorted(set(parents.tolist())),
        pair_invariant='same prefix through eligible parent; change next edge to latent; same suffix edges',
        parent_counts={str(x):int((parents[slots]==x).sum()) for x in np.unique(parents)},
        quantity='unnormalized q_i = g_i^T sum_j K_ij g_j; encoder parameters through post-block-1 boundary',
        host=socket.gethostname(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (a.out_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':main()
