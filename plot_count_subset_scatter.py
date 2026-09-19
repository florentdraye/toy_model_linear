"""Plot ten-seed fixed-background q scatter versus latent count."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',nargs='+',type=Path);p.add_argument('--out-dir',required=True,type=Path)
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    raw=[];metadata=[]
    for root in a.roots:
        metadata.append(json.loads((root/'metadata.json').read_text()))
        with (root/'scatter.csv').open() as f:
            for r in csv.DictReader(f):
                raw.append(dict(seed=int(r['seed']),step=int(r['step']),gain=float(r['gain']),
                    eval_id=int(r['eval_id']),count=int(r['count']),subset=int(r['subset']),
                    q_sum=float(r['q_sum']),q_extra=float(r['q_extra'])))
    if len(metadata)!=10 or not all(m['complete'] for m in metadata):raise ValueError('need ten complete seeds')
    grouped=defaultdict(list)
    for r in raw:grouped[(r['count'],r['subset'])].append(r)
    points=[]
    for (count,subset),z in sorted(grouped.items()):
        points.append(dict(count=count,subset=subset,q_sum=np.mean([r['q_sum'] for r in z]),
                           q_extra=np.mean([r['q_extra'] for r in z]),samples=len(z)))
    with (a.out_dir/'subset_points.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,points[0]);w.writeheader();w.writerows(points)
    by_count=defaultdict(list)
    for r in points:by_count[r['count']].append(r)
    summary=[]
    for count,z in sorted(by_count.items()):
        summary.append(dict(count=count,n_subsets=len(z),q_sum_mean=np.mean([r['q_sum'] for r in z]),
            q_sum_sd=np.std([r['q_sum'] for r in z]),q_extra_mean=np.mean([r['q_extra'] for r in z]),
            q_extra_sd=np.std([r['q_extra'] for r in z])))
    with (a.out_dir/'count_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,summary[0]);w.writeheader();w.writerows(summary)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),sharex=True)
    rng=np.random.default_rng(812)
    for ax,field,ylabel in ((axes[0],'q_sum','Full batch transfer  gᵢᵀ ΣⱼKᵢⱼgⱼ'),
                            (axes[1],'q_extra','Extra transfer relative to count 0')):
        for count,z in sorted(by_count.items()):
            y=np.array([r[field] for r in z]);x=count+rng.uniform(-.55,.55,len(y))
            ax.scatter(x,y,s=8,color='#31688e',alpha=.22,edgecolors='none')
        x=np.array([r['count'] for r in summary]);mean=np.array([r[field+'_mean'] for r in summary])
        ax.plot(x,mean,color='#440154',lw=2.1,label='mean over subsets')
        ax.axvline(metadata[0]['natural_expected_count'],color='#d95f02',ls='--',lw=1.5,
                   label=f"natural expectation = {metadata[0]['natural_expected_count']:.1f}")
        ax.axhline(0,color='.5',lw=.7);ax.grid(alpha=.18);ax.set_xlabel('Number of latent-78 examples in fixed batch')
        ax.set_ylabel(ylabel);ax.legend(frameon=False)
    gain=np.mean([m['gain'] for m in metadata]);steps=[m['step'] for m in metadata]
    fig.suptitle(f"Ordinary-frequency latent 78 — fixed n=2,048 eligible-parent contexts\n"
                 f"128 subsets per count; 10-seed mean at gain ≈ {gain:.3f} (steps {min(steps)}–{max(steps)})")
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(a.out_dir/f'count_subset_scatter.{suffix}',dpi=180 if suffix=='png' else None,bbox_inches='tight')
    natural=min(summary,key=lambda r:abs(r['count']-metadata[0]['natural_expected_count']))
    result=dict(latent=78,frequency=metadata[0]['frequency'],natural_expected_count=metadata[0]['natural_expected_count'],
        displayed_natural_count=natural['count'],mean_gain=gain,steps=steps,
        natural_q_extra_mean=natural['q_extra_mean'],natural_q_extra_sd_across_subsets=natural['q_extra_sd'],
        eligible_parents=metadata[0]['eligible_parents'])
    (a.out_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
