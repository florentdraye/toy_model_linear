"""Measure count transfer, context variation, contribution agreement, and alignment.

One job handles one latent and one raw optimized16 model seed. Heavy execution
belongs on an H100 compute node.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ModelConfig
from src.context_transfer import (cosine_rows, encode, encoder_parameters,
                                  evaluation_state, source_gradient,
                                  subtract_step, task_loss, transfer)
from src.data import enumerate_paths
from src.emergence import from_hidden
from src.graph import Graph
from src.model import ToyTransformer


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def tree_difference(a, b):
    return type(a)((name, a[name] - b[name]) for name in a)


def losses(model, params, edges, labels):
    return F.cross_entropy(from_hidden(model, encode(model, params, edges), 1).float(),
                           labels, reduction="none")


def matched_batches(pair_ids, n, counts, backgrounds, repeats, seed):
    rng = np.random.default_rng(seed)
    result = {}
    for bg in range(backgrounds):
        slots = rng.choice(len(pair_ids), n, replace=False)
        off = pair_ids[slots, 0].copy(); on = pair_ids[slots, 1].copy()
        for count in counts:
            for repeat in range(repeats):
                chosen = (np.empty(0, dtype=np.int64) if count == 0 else
                          rng.choice(n, count, replace=False))
                ids = off.copy(); ids[chosen] = on[chosen]
                result[bg, count, repeat] = (ids, chosen)
    return result


def no_shared_ids(train_ids, nodes, eval_id, number, rng):
    # Internal graph nodes are the active latent factors; root/output excluded.
    target = nodes[eval_id, 1:-1]
    candidates = train_ids[(nodes[train_ids, 1:-1, None] == target[None, None, :])
                           .any((1, 2)).logical_not()]
    if len(candidates) < number:
        raise ValueError("insufficient no-shared-latent controls")
    return candidates[torch.as_tensor(rng.choice(len(candidates), number, replace=False))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--latent", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--counts", type=int, nargs="+", default=[0, 16, 32, 64, 96, 128])
    p.add_argument("--backgrounds", type=int, default=3)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--eval-points", type=int, default=4)
    p.add_argument("--time-every", type=int, default=500)
    p.add_argument("--agreement-bank", type=int, default=8)
    p.add_argument("--eta", type=float, nargs="+", default=[0.01, 0.005])
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("submit this measurement to a GPU compute node")
    if a.out_dir.exists():
        raise FileExistsError(a.out_dir)
    if sorted(set(a.counts)) != a.counts or a.counts[0] != 0 or a.counts[-1] > a.batch_size:
        raise ValueError("counts must be unique, sorted, start at zero, and fit the batch")
    if not 0 < a.agreement_bank * 2 <= a.batch_size:
        raise ValueError("invalid agreement bank size")
    torch.set_num_threads(4); torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    source = json.loads((a.run / "history.json").read_text())
    if not source["complete"] or a.latent not in source["latents"]:
        raise ValueError("need a completed optimized16 source run and tracked latent")
    li = source["latents"].index(a.latent)
    if source["config"]["steer_depth"] != 1:
        raise ValueError("expected post-block-1 steering")
    paths = enumerate_paths(Graph.load(a.run / "graph.pt"))
    banks = torch.load(a.run / "banks.pt", weights_only=True)
    pair_ids = banks["fit"][li].numpy()
    eval_ids = banks["test"][li, :a.eval_points, 1]
    train_ids = banks["train"]
    edges, labels, nodes = paths["edge_seqs"], paths["labels"], paths["nodes"]
    eval_edges, eval_labels = edges[eval_ids].to(device), labels[eval_ids].to(device)
    rng = np.random.default_rng(19001 + 1009 * source["config"]["seed"] + a.latent)
    fixed = matched_batches(pair_ids, a.batch_size, a.counts, a.backgrounds, a.repeats,
                            21001 + a.latent)
    fixed_m = min(a.counts, key=lambda x: abs(x - a.batch_size // 2))
    target_reference = torch.as_tensor(pair_ids[:a.agreement_bank, 1])
    target_measure = torch.as_tensor(pair_ids[a.agreement_bank:2*a.agreement_bank, 1])
    controls = {int(eid): no_shared_ids(train_ids, nodes, int(eid), 2*a.agreement_bank,
                                        rng) for eid in eval_ids}

    trajectory = source["history"]
    stages = []
    for name, target in (("before", .1), ("during", .5), ("after", .9)):
        row = min(trajectory, key=lambda r: abs(r["gain_raw"][li] - target))
        stages.append({"stage": name, "target_gain": target, "step": row["step"],
                       "gain": row["gain_raw"][li]})
    time_steps = sorted({r["step"] for r in trajectory if r["step"] % a.time_every == 0}
                        | {s["step"] for s in stages})
    stage_by_step = {s["step"]: s for s in stages}

    a.out_dir.mkdir(parents=True)
    metadata = {
        "complete": False, "seed": source["config"]["seed"], "latent": a.latent,
        "latent_index": li, "frequency": source["frequency"][li], "run": str(a.run),
        "boundary": "complete post-block-1 residual [6,128]", "encoder_parameters":
        "token/position embeddings and transformer block 1", "downstream": "frozen blocks 2-6 and head",
        "task_loss": "100-class endpoint cross entropy", "batch_size": a.batch_size,
        "counts": a.counts, "fixed_m": fixed_m, "backgrounds": a.backgrounds,
        "repeats": a.repeats, "eval_ids": eval_ids.tolist(), "eta": a.eta,
        "stages": stages, "time_steps": time_steps, "paired_source_rule":
        "switch absent/present training paths with identical suffix edge choices",
        "control_rule": "training paths sharing no internal graph node with the evaluation path",
        "membership_sha256": hashlib.sha256(np.concatenate([v[0] for v in fixed.values()]).tobytes()).hexdigest(),
        "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (a.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    model = ToyTransformer(ModelConfig(**source["model_config"])).to(device).eval()
    count_rows, variation_rows, agreement_rows, alignment_rows = [], [], [], []
    start = time.time()

    for step in time_steps:
        model.load_state_dict(torch.load(a.run / "models" / f"step{step:06d}.pt",
                                         map_location="cpu", weights_only=True))
        params = encoder_parameters(model)
        before_loss, eval_g, _ = evaluation_state(model, params, eval_edges, eval_labels)

        # Fixed-count subset variation and useful alignment through time.
        for bg in range(a.backgrounds):
            zero_ids, _ = fixed[bg, 0, 0]
            zedges = edges[torch.as_tensor(zero_ids)].to(device)
            zlabels = labels[torch.as_tensor(zero_ids)].to(device)
            zero_grad = source_gradient(model, params, zedges, zlabels)
            for repeat in range(a.repeats):
                ids, chosen = fixed[bg, fixed_m, repeat]
                bedges = edges[torch.as_tensor(ids)].to(device)
                blabels = labels[torch.as_tensor(ids)].to(device)
                batch_grad = source_gradient(model, params, bedges, blabels)
                diff = tree_difference(batch_grad, zero_grad)
                td = transfer(model, params, eval_edges, diff)
                q = (eval_g.double() * td.double()).flatten(1).sum(1)
                tb = transfer(model, params, eval_edges, batch_grad)
                cos, dot, gn, tn = cosine_rows(eval_g, tb)
                for ei, eid in enumerate(eval_ids.tolist()):
                    variation_rows.append(dict(seed=metadata["seed"], step=step, latent=a.latent,
                        eval_id=eid, background=bg, subset=repeat, m=fixed_m, q=float(q[ei]),
                        subset_slots=";".join(map(str, chosen.tolist()))))
                    alignment_rows.append(dict(seed=metadata["seed"], step=step, latent=a.latent,
                        eval_id=eid, source="shared_latent_batch", background=bg, subset=repeat,
                        cosine=float(cos[ei]), dot=float(dot[ei]), g_norm=float(gn[ei]),
                        transfer_norm=float(tn[ei])))

        # Individual-contribution agreement. Reference and measurement banks are disjoint.
        target_ref_grads = [source_gradient(model, params, edges[j:j+1].to(device),
                            labels[j:j+1].to(device)) for j in target_reference.tolist()]
        target_measure_grads = [source_gradient(model, params, edges[j:j+1].to(device),
                                labels[j:j+1].to(device)) for j in target_measure.tolist()]
        for ei, eid in enumerate(eval_ids.tolist()):
            one_edge = eval_edges[ei:ei+1]
            tref = torch.stack([transfer(model, params, one_edge, d)[0] for d in target_ref_grads]).mean(0)
            control_ids = controls[eid]
            control_ref_grads = [source_gradient(model, params, edges[j:j+1].to(device),
                                 labels[j:j+1].to(device)) for j in control_ids[:a.agreement_bank].tolist()]
            control_measure_grads = [source_gradient(model, params, edges[j:j+1].to(device),
                                     labels[j:j+1].to(device)) for j in control_ids[a.agreement_bank:].tolist()]
            cref = torch.stack([transfer(model, params, one_edge, d)[0] for d in control_ref_grads]).mean(0)
            for group, ids0, grads0, ref in (("shared_latent", target_measure.tolist(), target_measure_grads, tref),
                                             ("no_shared_latent", control_ids[a.agreement_bank:].tolist(),
                                              control_measure_grads, cref)):
                for source_id, direction in zip(ids0, grads0):
                    tij = transfer(model, params, one_edge, direction)[0]
                    cos, dot, tn, mn = cosine_rows(tij[None], ref[None])
                    agreement_rows.append(dict(seed=metadata["seed"], step=step, latent=a.latent,
                        eval_id=eid, source_id=source_id, source=group, cosine=float(cos[0]),
                        dot=float(dot[0]), transfer_norm=float(tn[0]), mean_norm=float(mn[0])))

            # A genuinely no-shared-latent batch control for useful alignment.
            cids = control_ids[:a.agreement_bank]
            cg = source_gradient(model, params, edges[cids].to(device), labels[cids].to(device))
            ct = transfer(model, params, one_edge, cg)
            cc, cd, cgn, ctn = cosine_rows(eval_g[ei:ei+1], ct)
            alignment_rows.append(dict(seed=metadata["seed"], step=step, latent=a.latent,
                eval_id=eid, source="no_shared_latent", background=-1, subset=-1,
                cosine=float(cc[0]), dot=float(cd[0]), g_norm=float(cgn[0]),
                transfer_norm=float(ctn[0])))

        # Actual and first-order transfer at the three acquisition stages.
        if step in stage_by_step:
            stage = stage_by_step[step]
            for bg in range(a.backgrounds):
                for count in a.counts:
                    for repeat in range(a.repeats):
                        ids, chosen = fixed[bg, count, repeat]
                        ids = torch.as_tensor(ids)
                        direction = source_gradient(model, params, edges[ids].to(device), labels[ids].to(device))
                        tij = transfer(model, params, eval_edges, direction)
                        contraction = (eval_g.double() * tij.double()).flatten(1).sum(1)
                        for eta in a.eta:
                            actual = before_loss - losses(model, subtract_step(params, direction, eta),
                                                          eval_edges, eval_labels).detach()
                            for ei, eid in enumerate(eval_ids.tolist()):
                                count_rows.append(dict(seed=metadata["seed"], step=step,
                                    stage=stage["stage"], gain=stage["gain"], latent=a.latent,
                                    eval_id=eid, count=count, background=bg, repeat=repeat, eta=eta,
                                    actual_loss_decrease=float(actual[ei]),
                                    predicted_loss_decrease=float(eta * contraction[ei]),
                                    source_gradient_norm=float(sum(x.double().square().sum() for x in direction.values()).sqrt()),
                                    subset_slots=";".join(map(str, chosen.tolist()))))

        print(json.dumps({"latent": a.latent, "step": step,
                          "elapsed_seconds": round(time.time() - start, 1)}), flush=True)

    write_csv(a.out_dir / "count_transfer.csv", count_rows)
    write_csv(a.out_dir / "context_variation.csv", variation_rows)
    write_csv(a.out_dir / "contribution_agreement.csv", agreement_rows)
    write_csv(a.out_dir / "useful_alignment.csv", alignment_rows)
    metadata["complete"] = True; metadata["elapsed_seconds"] = round(time.time() - start, 2)
    (a.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"complete": True, "latent": a.latent,
                      "elapsed_seconds": metadata["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
