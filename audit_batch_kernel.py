"""Validate downloaded kernel/count artifacts and summarize conditional spread."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from src.batch_kernel import sample_batches


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--banks',type=Path,required=True)
    a=p.parse_args()
    trajectory=json.loads(a.trajectory.read_text())
    banks=torch.load(a.banks,weights_only=True)
    allowed=np.zeros(int(max(banks['train'].max(),banks['test_support'].max()))+1,dtype=bool)
    allowed[banks['train'].numpy()]=True
    ref_choices=torch.randint(banks['test'].shape[1],(len(trajectory['latents']),),generator=torch.Generator().manual_seed(7141))
    rows=[];hashes=set();zs=[];errors=[]
    for j,k in enumerate(trajectory['latents']):
        path=a.run/f'latent_{k}'
        h=json.loads((path/'history.json').read_text())
        assert h['complete'] and len(h['stages'])==3
        assert h['reference_id']==int(banks['test'][j,ref_choices[j],1])
        assert not allowed[h['reference_id']]
        with np.load(path/'pool.npz') as f:pool={key:f[key] for key in f.files}
        assert allowed[pool['ids']].all() and len(np.unique(pool['ids']))==102400
        digest=hashlib.sha256(pool['ids'].tobytes()).hexdigest()
        assert digest==h['pool_sha256'];hashes.add(digest)
        assert all(np.sum(pool['latent']==c)==1024 for c in h['classes'])
        with np.load(path/'scatter.npz') as f:data={key:f[key] for key in f.files}
        assert data['responses'].shape==(10,64,3,2)
        assert (data['class_counts'].sum(1)==16384).all()
        np.testing.assert_array_equal(data['class_counts'][:,h['classes'].index(k)],data['target_counts'])
        products=[]
        for stage in h['stages']:
            selected=min(trajectory['history'],key=lambda r:abs(r['gain_raw'][j]-stage['target_gain']))
            assert selected['step']==stage['step'] and selected['gain_raw'][j]==stage['actual_gain']
            with np.load(path/f"kernel_{stage['name']}.npz") as f:products.append(f['products'])
            errors.extend(stage['aggregate_check']['relative_error'])
        products=np.stack(products)
        assert products.shape==(3,102400,2) and np.isfinite(products).all()
        means=np.stack([products[:,pool['latent']==c,:].mean(1) for c in h['classes']],axis=1)
        variances=np.stack([products[:,pool['latent']==c,:].var(1,ddof=0) for c in h['classes']],axis=1)
        expected=np.einsum('cl,slq->csq',data['class_counts'],means)
        predicted_sd=np.sqrt(np.einsum('cl,slq->csq',data['class_counts'],variances))
        np.testing.assert_allclose(data['expected'],expected,rtol=1e-11,atol=1e-7)
        np.testing.assert_allclose(data['predicted_sd'],predicted_sd,rtol=1e-11,atol=1e-7)
        z=(data['responses'].mean(1)-expected)/(predicted_sd/8)
        zs.extend(z[np.isfinite(z)].tolist())
        if j==0:
            replay=sample_batches(products,pool['latent'],h['classes'],h['probabilities'],k,
                                  h['counts'],h['batch_size'],h['repeats'],h['sampling_seed'])
            np.testing.assert_allclose(replay['responses'],data['responses'],rtol=1e-12,atol=1e-7)
        for s,stage in enumerate(h['stages']):
            for col,name in enumerate(['logit','loss']):
                sd=data['responses'][:,:,s,col].std(1,ddof=1).mean()
                span=np.ptp(expected[:,s,col])
                rows.append(dict(latent=k,stage=stage['name'],projection=name,within_sd_over_span=float(sd/span),
                                  scaled_within_sd=float(sd/128),reference_ce=stage['reference']['ce_loss']))
    assert len(hashes)==1
    aggregates={}
    for projection in ['logit','loss']:
        aggregates[projection]={}
        for stage in ['before','midpoint','after']:
            subset=[r for r in rows if r['projection']==projection and r['stage']==stage]
            aggregates[projection][stage]={key:float(np.median([r[key] for r in subset]))
                  for key in ['within_sd_over_span','scaled_within_sd','reference_ce']}
        values={(r['latent'],r['stage']):r['within_sd_over_span'] for r in rows if r['projection']==projection}
        aggregates[projection]['after_less_than_before']=sum(values[k,'after']<values[k,'before'] for k in trajectory['latents'])
        aggregates[projection]['after_less_than_midpoint']=sum(values[k,'after']<values[k,'midpoint'] for k in trajectory['latents'])
    z=np.asarray(zs)
    report=dict(complete=True,latents=32,stages=96,point_kernel_responses=32*3*102400,
                scatter_points_per_projection=32*3*10*64,shared_pool_sha256=hashes.pop(),
                max_aggregate_relative_error=float(max(errors)),
                mc_mean_error_z_rms=float(np.sqrt(np.mean(z*z))),
                mc_mean_error_fraction_within_2se=float(np.mean(abs(z)<2)),
                checks='reference/train split, unique fixed pools, full counts, stage selection, all moments, first-latent batch replay passed',
                aggregates=aggregates)
    (a.run/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
