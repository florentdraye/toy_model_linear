"""Raw batch-count scatters, one three-stage figure per latent and projection."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def panel(ax, h, data, stage, projection):
    x=data['target_counts']; y=data['responses'][:,:,stage,projection]
    ax.scatter(np.repeat(x,y.shape[1]),y.reshape(-1),s=9,alpha=.3,color='#326b9a',edgecolors='none')
    ax.plot(x,data['expected'][:,stage,projection],color='#a95d21',lw=1.2,label='Conditional pool mean')
    s=h['stages'][stage]
    ax.set_title(f"{s['name'].capitalize()} · step {s['step']:,}\nlatent gain {s['actual_gain']:.3f}",fontsize=10)
    ax.set_xlabel('Target-latent count in batch')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    ax.grid(alpha=.15)
    ax.spines[['top','right']].set_visible(False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--out-dir',type=Path)
    p.add_argument('--expected-latents',type=int)
    a=p.parse_args()
    out=a.out_dir or a.run/'figures';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in a.run.glob('latent_*/history.json'):
        h=json.loads(path.read_text())
        if not h['complete']:continue
        with np.load(path.parent/'scatter.npz') as f:data={k:f[k] for k in f.files}
        if not np.isfinite(data['responses']).all():raise ValueError('nonfinite responses')
        if not (data['class_counts'].sum(1)==h['batch_size']).all():raise ValueError('batch size mismatch')
        if not np.array_equal(data['class_counts'][:,h['classes'].index(h['latent'])],data['target_counts']):raise ValueError('target count mismatch')
        records.append((h,data))
    records.sort(key=lambda v:-v[0]['frequency'])
    if not records or (a.expected_latents is not None and len(records)!=a.expected_latents):
        raise ValueError(f'found {len(records)} complete latents')
    summaries=[];points=[]
    for projection,slug in enumerate(['logit','loss']):
        with PdfPages(out/f'all_latents_{slug}.pdf') as pdf:
            for h,data in records:
                fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
                for stage,ax in enumerate(axes):panel(ax,h,data,stage,projection)
                ylabel=(r'$[\sum_j K_{ij}g_j]_{y_i}$' if projection==0 else r'$g_i^T\sum_j K_{ij}g_j$')
                axes[0].set_ylabel(ylabel)
                fig.suptitle(f"Latent {h['latent']} · frequency {h['frequency']:.4f} · reference path {h['reference_id']}\n"
                    f"EMA 0.95 · batch {h['batch_size']:,} · {h['repeats']} batches/count · orange: conditional pool mean",fontsize=11)
                fig.savefig(out/f"latent_{h['latent']}_{slug}.png",dpi=180)
                pdf.savefig(fig);plt.close(fig)
                for stage,s in enumerate(h['stages']):
                    y=data['responses'][:,:,stage,projection]
                    expected=data['expected'][:,stage,projection]
                    span=float(np.ptp(expected))
                    sd=y.std(1,ddof=1)
                    summaries.append(dict(latent=h['latent'],frequency=h['frequency'],projection=slug,
                        stage=s['name'],step=s['step'],gain=s['actual_gain'],reference_ce=s['reference']['ce_loss'],
                        conditional_mean_span=span,mean_within_count_sd=float(sd.mean()),
                        within_sd_over_count_span=float(sd.mean()/span) if span>0 else None,
                        mean_sd_of_batch_mean_times_sqrt_batch=float(sd.mean()/np.sqrt(h['batch_size'])),
                        predicted_mean_within_count_sd=float(data['predicted_sd'][:,stage,projection].mean())))
                    for ci,n in enumerate(data['target_counts']):
                        for b,value in enumerate(y[ci]):
                            points.append(dict(latent=h['latent'],projection=slug,stage=s['name'],step=s['step'],
                                gain=s['actual_gain'],count=int(n),replicate=b,response=float(value),
                                batch_mean_response=float(value/h['batch_size'])))
        # Frequency-selected examples, not selected for favorable response plots.
        chosen=np.linspace(0,len(records)-1,min(8,len(records))).round().astype(int)
        fig,axes=plt.subplots(len(chosen),3,figsize=(13,3*len(chosen)),squeeze=False,layout='constrained')
        for row,index in enumerate(chosen):
            h,data=records[index]
            for stage,ax in enumerate(axes[row]):panel(ax,h,data,stage,projection)
            axes[row,0].set_ylabel(f"Latent {h['latent']}\n"+(r'$[\sum K g]_{y_i}$' if projection==0 else r'$g_i^T\sum K g$'))
        fig.suptitle(f"{len(chosen)} frequency-spaced latents · EMA 0.95 · raw {slug} response sums\n"
                     'Same reference across stages; 16,384 examples/batch; 64 batches/count; orange: conditional pool mean',fontsize=12)
        fig.savefig(out/f'overview_{slug}.png',dpi=170);plt.close(fig)
    for filename,rows in [('summary.csv',summaries),('scatter_points.csv',points)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (out/'figure_notes.txt').write_text(
        'Every dot is a sampled batch, not an independent model run. Y is an unnormalized sum.\n'
        'No smoothing, outcome selection, clipping or synthetic residuals.\n'
        'The target count replaces other latent counts at fixed batch size. Non-target proportions\n'
        'follow the training law, rounded deterministically; all class counts are fixed at each x.\n'
        'Examples are sampled with replacement from fixed, distinct training-pool strata.\n'
        'The pool-conditional expectation is near-affine by construction: a straight line alone\n'
        'does not establish count sufficiency. Inspect within-count spread and its batch-size scaling.\n'
        'The kernel uses all model parameters and positive cross-entropy logit gradients.\n'
        'Negative learning_rate/batch_size times the sums gives first-order SGD changes.\n'
        'This is not an Adam update or a measured finite-step loss change.\n')
    print(f'{len(records)} latents; figures, raw scatter CSV and statistics: {out}')


if __name__=='__main__':main()
