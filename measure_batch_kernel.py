"""Kernel response versus exact latent counts, using frozen EMA snapshots on GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch
from src.batch_kernel import reference_directions, projected_contributions, validate_aggregate, sample_batches
from src.config import ModelConfig
from src.data import enumerate_paths
from src.graph import Graph
from src.model import ToyTransformer
from train_emergence import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run',type=Path,required=True)
    p.add_argument('--model-run',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--latent',type=int,required=True)
    p.add_argument('--pool-per-class',type=int,default=1024)
    p.add_argument('--microbatch',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=16384)
    p.add_argument('--repeats',type=int,default=64)
    p.add_argument('--counts',type=int,nargs='+',default=[0,128,256,512,1024,2048,4096,8192,12288,16384])
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Submit to a GPU compute node')
    if min(a.pool_per_class,a.microbatch,a.batch_size,a.repeats)<2:raise ValueError('sizes must be >=2')
    if sorted(set(a.counts))!=a.counts or a.counts[0]<0 or a.counts[-1]>a.batch_size:raise ValueError('invalid counts')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(7091)
    source=json.loads((a.training_run/'history.json').read_text())
    trajectory=json.loads((a.model_run/'history.json').read_text())
    if not trajectory['complete']:raise ValueError('need completed trajectory')
    if source['evaluation_bank_sha256']!=trajectory['evaluation_bank_sha256']:raise ValueError('bank mismatch')
    index=source['latents'].index(a.latent)
    paths=enumerate_paths(Graph.load(a.training_run/'graph.pt'))
    banks=torch.load(a.training_run/'banks.pt',weights_only=True)
    train=banks['train']
    graph_layer=source['config']['graph_layer']
    classes=source['classes']
    rng=torch.Generator().manual_seed(7139)
    ids=[]
    train_latents=paths['nodes'][train,graph_layer]
    for k in classes:
        ix=train[train_latents==k]
        if len(ix)<a.pool_per_class:raise ValueError(f'insufficient unique pool support for {k}')
        ids.append(ix[torch.randperm(len(ix),generator=rng)[:a.pool_per_class]])
    ids=torch.cat(ids)
    pool_classes=paths['nodes'][ids,graph_layer].numpy()
    ref_rng=torch.Generator().manual_seed(7141)
    ref_choices=torch.randint(banks['test'].shape[1],(len(source['latents']),),generator=ref_rng)
    ref_id=int(banks['test'][index,ref_choices[index],1])
    if bool((train==ref_id).any()) or int(paths['nodes'][ref_id,graph_layer])!=a.latent:raise ValueError('invalid reference')
    ref_edges=paths['edge_seqs'][ref_id:ref_id+1].cuda()
    ref_label=paths['labels'][ref_id].cuda()
    edges=paths['edge_seqs'][ids].cuda()
    labels=paths['labels'][ids].cuda()
    stages=[]
    for name,target in [('before',.1),('midpoint',.5),('after',.9)]:
        row=min(trajectory['history'],key=lambda r:abs(r['gain_raw'][index]-target))
        stages.append(dict(name=name,target_gain=target,step=row['step'],actual_gain=row['gain_raw'][index]))
    if not stages[0]['step']<stages[1]['step']<stages[2]['step']:raise ValueError('stages not chronological')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.out_dir/'pool.npz',ids=ids.numpy(),latent=pool_classes,
                        endpoints=paths['labels'][ids].numpy(),reference_edges=ref_edges.cpu().numpy())
    result=dict(complete=False,latent=a.latent,frequency=source['frequency'][index],reference_id=ref_id,
                source_run=str(a.training_run),model_run=str(a.model_run),model_transform=trajectory['model_transform'],
                model_config=source['model_config'],seed=source['config']['seed'],classes=classes,
                probabilities=source['probabilities'],pool_per_class=a.pool_per_class,pool_size=len(ids),
                pool_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),pool_seed=7139,reference_seed=7141,
                sampling_seed=8831+a.latent,batch_size=a.batch_size,repeats=a.repeats,counts=a.counts,
                sample_method='PCG64; with replacement within each latent stratum; exact full count vector per x',
                kernel='logit NTK J_i J_j^T over all model parameters; Euclidean metric',
                g_definition='positive cross-entropy logit gradient: softmax(z)-one_hot(y)',
                projections=['correct_endpoint_logit','reference_loss_contraction'],
                optimization_interpretation='minus learning_rate/batch_size times sum gives first-order SGD response; excludes Adam, clipping and weight decay',
                dtype='FP32 model/JVP; FP64 softmax, contractions and batch sums; math SDPA',
                host=socket.gethostname(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),stages=[])
    atomic_json(a.out_dir/'history.json',result)
    print(json.dumps({k:result[k] for k in ['latent','reference_id','pool_size','host','gpu']}),flush=True)
    model=ToyTransformer(ModelConfig(**source['model_config'])).cuda().eval()
    all_products=[]
    start=time.time()
    for stage in stages:
        model.load_state_dict(torch.load(Path(trajectory['model_directory'])/f"step{stage['step']:06d}.pt",map_location='cpu',weights_only=True))
        directions,reference=reference_directions(model,ref_edges,ref_label)
        products=projected_contributions(model,edges,labels,directions,a.microbatch)
        # Cross-check the opposite product ordering on examples spanning all classes.
        check_ids=torch.linspace(0,len(ids)-1,32).long()
        check=validate_aggregate(model,ref_edges,ref_label,edges[check_ids.cuda()],labels[check_ids.cuda()],products[check_ids])
        norms=[float(sum(v.double().square().sum() for v in d.values()).sqrt()) for d in directions]
        row={**stage,'reference':reference,'reference_gradient_norms':norms,'aggregate_check':check,
             'elapsed_seconds':round(time.time()-start,2)}
        np.savez_compressed(a.out_dir/f"kernel_{stage['name']}.npz",products=products.numpy())
        result['stages'].append(row)
        all_products.append(products.numpy())
        atomic_json(a.out_dir/'history.json',result)
        print(json.dumps(row),flush=True)
        del directions,products
    sampled=sample_batches(np.stack(all_products),pool_classes,classes,source['probabilities'],a.latent,
                           a.counts,a.batch_size,a.repeats,result['sampling_seed'])
    np.savez_compressed(a.out_dir/'scatter.npz',**sampled)
    result['complete']=True
    result['elapsed_seconds']=round(time.time()-start,2)
    atomic_json(a.out_dir/'history.json',result)
    print(json.dumps(dict(complete=True,latent=a.latent,elapsed=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
