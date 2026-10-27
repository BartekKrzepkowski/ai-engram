"""ReCAST on TOFU forget10 (Llama-3.2-1B-Instruct), scored like Table 3 of arXiv 2606.14997.

One run = one variant at one shrink floor:

- ``unrestricted``: retain-aligned row shrink ``W* = s * W`` of every decoder-block linear
  weight, then phase two trains the whole model;
- ``restricted_semu``: the same proposal projected into the forget-gradient SVD subspace
  (``W* = W + P(s * W - W)``), each target replaced by a SEMU adapter ``anchor + U A Vh``, and
  phase two trains the cores ``A`` plus the groups without an adapter named by ``--phase2_free``
  (``none`` = only the cores, as Classification);
- ``restricted_shrink`` (extension, user request 2026-10-22): the restricted warm start written into
  the native weights, no adapter, and phase two trains every parameter as ``unrestricted``.

Restricted variants can soften the warm start along the input directions retain uses
(``--retain_input_filter ridge``, extension): the projected delta is right-multiplied by
``mu (C_r + mu I)^-1``, C_r the retain input covariance over answer tokens at Init Fine. (A), or
build each restriction basis from the forget gradient whitened by C_r (``--restriction_whitening
retain_input``, ``recast_core.whitened_svd_basis``) (C). The edit budget is the EVR rank, an explicit
rank multiple of it, or a forget-output target per target (``--restriction_forget_output_target``).
``--component_selection`` (extension, signed shrink) keeps, removes or rescales the Frobenius-orthogonal
components ``D_j`` of each projected update by the sign of their first-order forget contribution
``c_j = <G_f, D_j>``, with a random and a norm-matched control; ``--component_diagnostics yes`` reports the
decomposition and the actual against the predicted forget / retain loss change of the warm start.
``--epochs 0`` evaluates the warm-started model without phase two, and ``--eval_trajectory nll`` tracks
the answer-token NLL and exact memorization of forget, retain (the 400 Model Utility items), real authors
and world facts before phase two and after every epoch (extensions).

The evaluation also keeps what ``compute_full_tofu`` computes and discards (``tofu_capture.py``):
the nine sub-metrics behind Model Utility (``_model_utility_subs``) and the generations behind the
four ROUGE scores (``generations.json``); no aggregate changes.

Phase two uses retain data only (no forget loss; user decision 2026-10-22) with the
OpenUnlearning TOFU trainer recipe, through the Hugging Face Trainer. ``--retain_loss`` picks the
retain NLL, KL(Init Fine. || model) per non-padding position against a frozen copy of the starting
model, or NLL + ``--kl_weight`` x KL (extension, user request 2026-10-22). The unlearned model is
scored with engram.benchmarks._tofu_official against the Init Fine. and Retain raw metrics of
the Table 3 run, so its row is directly comparable with Table 3.

Usage (GH200 node, ai-engram-gh200 environment, offline cache): every argument is required.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time

from recast_core import (COMPONENT_SELECTIONS, TARGET_PROFILES, SemuAdapterLinear, apply_semu_adapters, change_report,
                         collect_input_covariances, collect_mean_gradients, drift_report, free_parameters,
                         frozen_checksums, mix_sources, ridge_input_filter, svd_basis,
                         parse_blocks,
                         parse_points,
                         parse_type_scales,
                         piecewise_linear,
                         scale_update, select_targets, snapshot_trainable, untie_output_embedding, warm_start)
from table3_tofu import COLUMNS, PAPER_TABLE3, Stage, git_info, to_jsonable
from tofu_capture import TofuCapture

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")


def parse_args():
    p = argparse.ArgumentParser(description="ReCAST on TOFU, scored like Table 3")
    # data and model
    p.add_argument("--base_id", required=True)
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--forget_split", required=True, help="e.g. forget10 (forget gradient)")
    p.add_argument("--retain_split", required=True, help="e.g. retain90 (retain gradient and phase two)")
    p.add_argument("--holdout_split", required=True, help="e.g. holdout10 (evaluation only)")
    p.add_argument("--train_dtype", required=True, choices=["bfloat16", "float32"],
                   help="weights during gradient collection, warm start and phase two")
    p.add_argument("--eval_dtype", required=True, choices=["bfloat16"], help="weights during evaluation (as the anchors)")
    p.add_argument("--attn_implementation", required=True)
    # ReCAST phase one
    p.add_argument("--variant", required=True, choices=["unrestricted", "restricted_semu", "restricted_shrink"],
                   help="restricted_shrink: restricted warm start, then phase two on every parameter (no adapter)")
    p.add_argument("--targets", required=True, choices=sorted(TARGET_PROFILES))
    p.add_argument("--target_blocks", required=True, help="'all' or decoder-block indices, e.g. 8-15 or 0,2,5-7")
    p.add_argument("--untie_lm_head", choices=["yes", "no"],
                   help="when the head is a target or trained: give it its own weight copy")
    p.add_argument("--train_lm_head", required=True, choices=["yes", "no"],
                   help="phase two trains the head outside any restriction (Classification fc rule)")
    p.add_argument("--s_min", type=float, required=True, help="shrink floor: s = max(cos(W_row, -G_r,row), s_min)")
    p.add_argument("--grad_batch_size", type=int, required=True)
    p.add_argument("--freeze_input_embedding", choices=["yes", "no"], help="unrestricted only")
    p.add_argument("--restriction_source", choices=["forget_only", "full_data"],
                   help="restricted only; full_data = example-weighted mean of forget and retain gradients")
    p.add_argument("--restriction_alignment_scope", choices=["full_space", "restriction_subspace"],
                   help="restricted only: align W with -G_r, or P(W) with -P(G_r) (Classification option)")
    p.add_argument("--shrink_layer_scaling", required=True, choices=["none", "subspace_norm", "subspace_norm_mean"],
                   help="subspace_norm (restricted, extension): per-target floor 1-(1-s_min)*r/max r, r=||P(W)||/||W||")
    p.add_argument("--restriction_evr", type=float, help="restricted only: EVR threshold of the forget-gradient SVD")
    p.add_argument("--restricted_head", choices=["unrestricted", "adapter"],
                   help="restricted with the head: 'unrestricted' = Classification --recast_train_fc_unrestricted")
    p.add_argument("--retain_input_filter", choices=["none", "ridge"],
                   help="restricted only (extension): right-multiply the warm-start delta by mu (C_r + mu I)^-1, "
                        "C_r the retain input covariance over answer tokens, mu = kappa tr(C_r) / d")
    p.add_argument("--retain_filter_kappa", type=float, help="ridge filter only: kappa")
    p.add_argument("--restriction_whitening", choices=["none", "retain_input"],
                   help="restricted only (extension): restriction bases from the SVD of G_f (C_r + eps I)^-1/2, "
                        "C_r the retain input covariance over answer tokens, eps = rho tr(C_r) / d")
    p.add_argument("--whitening_rho", type=float, help="retain_input whitening only: rho")
    p.add_argument("--restriction_rank_multiplier", type=float,
                   help="restricted only (extension): explicit edit budget, rank = round(m * r0) per target, r0 the "
                        "plain-basis rank at --restriction_evr; the same ranks for the plain and the whitened basis")
    p.add_argument("--restriction_representation",
                   choices=["none", "identity", "gradient", "whitened_gradient", "representation"],
                   help="restricted only (extension, basis D): input covectors from C_f v = gamma (C_r + eps I) v, "
                        "eps = rho tr(C_r) / d (--whitening_rho); output side identity (one-sided), gradient (G_f V), "
                        "whitened_gradient (the output side of C) or representation (W C W^T contrast); needs "
                        "--restriction_rank_multiplier")
    p.add_argument("--restriction_forget_output_target", type=float,
                   help="restricted only (extension): per target the smallest rank of its basis family whose "
                        "warm-start delta D changes forget outputs by tr(D C_f D^T) / tr(W C_f W^T) >= this value, "
                        "C_f the forget input covariance over answer tokens; replaces --restriction_rank_multiplier")
    p.add_argument("--component_selection", choices=list(COMPONENT_SELECTIONS),
                   help="restricted only (extension): apply sum_j alpha_j D_j of the components D_j of each projected "
                        "update, c_j = <G_f, D_j>: full (alpha 1, the SVD reconstruction), signed (1 if c_j > 0, else "
                        "0), random (as many ones as signed, drawn with --seed), uniform_matched (the original update "
                        "scaled to the signed update's Frobenius norm, per target); absent = the original update")
    p.add_argument("--component_diagnostics", choices=["yes"],
                   help="restricted only (extension): per-target components (sigma, c_j) to components.json, and the "
                        "forget / retain loss before and after the warm start against <G, Delta W>; with phase two "
                        "on native weights also after phase two: ||theta - theta_Init|| over the targets and over all "
                        "parameters, and <G, theta - theta_Init> with every parameter's Init Fine. mean gradient")
    p.add_argument("--warm_start_scale", type=float,
                   help="restricted only (extension): apply W_Init + t (W* - W_Init) with this t instead of W*, every "
                        "target from its Init weights; t in [0, 1 / (1 - s_min)] (s_min 0.3: t = 1 is W*)")
    p.add_argument("--warm_start_scale_target", choices=["holdout"],
                   help="restricted only (extension, gradient-free): choose t by bisection, with forward passes "
                        "only, so that the forget loss of W_Init + t (W* - W_Init) equals the Init Fine. loss on the "
                        "holdout split (authors the model never saw)")
    p.add_argument("--warm_start_scale_by_type",
                   help="restricted only (extension): --warm_start_scale per projection type, e.g. "
                        "down_proj=1,up_proj=0.714,o_proj=0.714, every type of --targets listed; with every row at the "
                        "floor (row alignment ~ 0) t is the per-type floor s = 1 - (1 - s_min) t")
    p.add_argument("--direction_weighting", choices=["contrast"],
                   help="restricted only (extension, gradient-free): shrink each input direction v_k of the basis by "
                        "s_k = 1 - (1 - s_min) w_k, w_k = forget/retain input-energy share ratio along v_k over its "
                        "mean (clipped so that s_k >= 0)")
    p.add_argument("--representation_mix", type=float,
                   help="basis D only: input-side forget signal (1 - beta) G_f^T G_f / tr + beta C_f / tr; "
                        "1 (default) = activations (D), 0 = gradient (the input side of C)")
    p.add_argument("--phase2_free",
                   help="restricted only: parameters without an adapter that phase two trains freely: none, all, "
                        "or a comma list of norms, attention, mlp (extension; Classification freezes them)")
    # phase two (OpenUnlearning TOFU recipe)
    p.add_argument("--epochs", type=float, required=True, help="0: evaluate the warm start, no phase two")
    p.add_argument("--eval_trajectory", choices=["nll"],
                   help="extension: before phase two and after every epoch, the answer-token NLL and exact "
                        "memorization (teacher-forced, as TOFU's metric_em) of forget, retain_perturbed, real "
                        "authors and world facts, plus the mean training loss of the epoch; random state restored, so "
                        "training is unchanged")
    p.add_argument("--epoch_length", required=True, choices=["forget", "retain"],
                   help="forget: an epoch is len(forget) random retain items (OpenUnlearning anchor=forget); "
                        "retain: an epoch is one pass over the retain split")
    p.add_argument("--per_device_batch_size", type=int, required=True)
    p.add_argument("--grad_accum", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight_decay", type=float, required=True)
    p.add_argument("--warmup_epochs", type=float, help="Hugging Face schedulers only")
    p.add_argument("--lr_scheduler", required=True,
                   help="a Hugging Face scheduler name, or piecewise_epochs (linear through --lr_points)")
    p.add_argument("--lr_points", help="piecewise_epochs only: epoch:lr pairs, e.g. 0:0,1:3e-5,5:1e-5,10:1e-5")
    p.add_argument("--optim", required=True)
    p.add_argument("--retain_loss", required=True, choices=["nll", "kl", "nll_kl"],
                   help="kl: KL(Init Fine. || model) per non-padding position of the retain batch")
    p.add_argument("--kl_weight", type=float, help="weight of the KL term (retain_loss kl or nll_kl)")
    p.add_argument("--max_grad_norm", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    # evaluation and bookkeeping
    p.add_argument("--eval_batch_size", type=int, required=True)
    p.add_argument("--anchors_raw", required=True, help="raw_metrics.json holding 'Init Fine.' and 'Retain'")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--save_model_dir", help="extension: save the model after phase two (train dtype) to this directory")
    p.add_argument("--run_name", required=True)
    p.add_argument("--wandb_mode", required=True, choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_project", required=True)
    p.add_argument("--wandb_tags", nargs="+", required=True)
    args = p.parse_args()
    with_head = TARGET_PROFILES[args.targets][1]
    restricted = args.variant != "unrestricted"
    piecewise = args.lr_scheduler == "piecewise_epochs"
    needed = {  # argument -> whether this configuration uses it
        "untie_lm_head": with_head or args.train_lm_head == "yes",
        "freeze_input_embedding": not restricted,
        "restriction_source": restricted,
        "restriction_evr": restricted,
        "restriction_alignment_scope": restricted,
        "restricted_head": restricted and with_head,
        "phase2_free": args.variant == "restricted_semu",
        "retain_input_filter": restricted,
        "retain_filter_kappa": args.retain_input_filter == "ridge",
        "restriction_whitening": restricted,
        "whitening_rho": args.restriction_whitening == "retain_input" or args.restriction_representation not in (None, "none"),
        "warmup_epochs": not piecewise,
        "lr_points": piecewise,
        "kl_weight": "kl" in args.retain_loss,
    }
    if args.restriction_rank_multiplier is not None and (not restricted or args.restriction_rank_multiplier <= 0):
        p.error("--restriction_rank_multiplier needs a restricted variant and a positive value")
    target = args.restriction_forget_output_target
    if target is not None and (not restricted or not 0.0 < target < 1.0 or args.restriction_rank_multiplier is not None
                               or args.retain_input_filter != "none" or args.shrink_layer_scaling != "none"
                               or args.restriction_alignment_scope != "full_space"):
        p.error("--restriction_forget_output_target needs a restricted variant, a value in (0, 1), no "
                "--restriction_rank_multiplier, --retain_input_filter none, --shrink_layer_scaling none and "
                "--restriction_alignment_scope full_space")
    if args.restriction_representation not in (None, "none") and (
            (args.restriction_rank_multiplier is None and target is None) or args.restriction_whitening != "none"):
        p.error("--restriction_representation needs --restriction_rank_multiplier or "
                "--restriction_forget_output_target, and --restriction_whitening none")
    if args.epochs < 0:
        p.error("--epochs must be non-negative (0 = no phase two)")
    if args.warm_start_scale is not None and (args.warm_start_scale_target is not None or not restricted
                                              or not 0.0 <= args.warm_start_scale
                                              <= (1.0 / (1.0 - args.s_min) if args.s_min < 1.0 else 0.0)):
        p.error("--warm_start_scale needs a restricted variant, no --warm_start_scale_target and a value in "
                "[0, 1 / (1 - s_min)]")
    if args.warm_start_scale_target is not None and (not restricted or args.s_min >= 1.0):
        p.error("--warm_start_scale_target needs a restricted variant and s_min < 1")
    if args.warm_start_scale_by_type is not None:
        try:
            by_type = parse_type_scales(args.warm_start_scale_by_type)
        except ValueError as e:
            p.error(str(e))
        cap = 1.0 / (1.0 - args.s_min) if args.s_min < 1.0 else 0.0
        if (not restricted or args.warm_start_scale is not None or args.warm_start_scale_target is not None
                or set(by_type) != set(TARGET_PROFILES[args.targets][0])
                or not all(0.0 <= t <= cap for t in by_type.values())):
            p.error("--warm_start_scale_by_type needs a restricted variant, no other scale argument, every projection "
                    f"type of --targets {args.targets} exactly once and values in [0, 1 / (1 - s_min)]")
    if args.direction_weighting is not None and (not restricted or args.retain_input_filter != "none"
                                                 or args.component_selection or args.component_diagnostics):
        p.error("--direction_weighting needs a restricted variant, --retain_input_filter none and no component "
                "selection or diagnostics")
    if (args.component_selection or args.component_diagnostics) and (
            not restricted or args.restriction_source != "forget_only" or args.retain_input_filter != "none"):
        p.error("--component_selection / --component_diagnostics need a restricted variant, --restriction_source "
                "forget_only (c_j uses the forget gradient) and --retain_input_filter none")
    if args.eval_trajectory and (args.epochs == 0 or args.epochs != int(args.epochs)):
        p.error("--eval_trajectory needs a positive whole number of --epochs")
    if args.representation_mix is not None and (args.restriction_representation in (None, "none")
                                                or not 0.0 <= args.representation_mix <= 1.0):
        p.error("--representation_mix needs --restriction_representation and a value in [0, 1]")
    missing = [k for k, use in needed.items() if use and getattr(args, k) is None]
    unused = [k for k, use in needed.items() if not use and getattr(args, k) is not None]
    if missing:
        p.error(f"{args.variant} with --targets {args.targets} needs --{', --'.join(missing)}")
    if unused:
        p.error(f"{args.variant} with --targets {args.targets} does not use --{', --'.join(unused)}; do not pass them")
    if (args.untie_lm_head == "no" and args.train_lm_head == "yes" and args.freeze_input_embedding == "yes"):
        p.error("a tied head is the input embedding: --train_lm_head yes contradicts --freeze_input_embedding yes")
    if args.shrink_layer_scaling != "none" and not restricted:
        p.error("--shrink_layer_scaling other than none needs the restricted variant")
    if piecewise:
        peak = max(y for _, y in parse_points(args.lr_points))
        if abs(peak - args.lr) > 1e-12 * max(1.0, args.lr):
            p.error(f"--lr must equal the peak of --lr_points ({peak:g}), got {args.lr:g}")
    return args


def summarize_by_projection(report):
    """Aggregate the per-target warm-start report over the seven projection types."""
    out = {}
    for proj in PROJECTIONS:
        rows = [v for k, v in report.items() if k.endswith(proj)]
        if not rows:
            continue
        agg = {"targets": len(rows),
               "shrink_mean": sum(r["shrink_mean"] for r in rows) / len(rows),
               "shrink_at_floor": sum(r["shrink_at_floor"] for r in rows) / len(rows),
               "alignment_mean": sum(r["alignment_mean"] for r in rows) / len(rows),
               "weight_rel_change_mean": sum(r["weight_rel_change"] for r in rows) / len(rows),
               "floor_min": min(r["floor"] for r in rows), "floor_max": max(r["floor"] for r in rows)}
        kept = [r["retain_filter_kept"] for r in rows if "retain_filter_kept" in r]
        if kept:
            agg.update({"retain_filter_kept_mean": sum(kept) / len(kept), "retain_filter_kept_min": min(kept)})
        if "rank" in rows[0]:
            ranks = sorted(r["rank"] for r in rows)
            agg.update({"rank_min": ranks[0], "rank_median": ranks[len(ranks) // 2], "rank_max": ranks[-1],
                        "total_rank": rows[0]["total_rank"], "inactive": sum(not r["active"] for r in rows),
                        "kept_fraction_of_proposal": sum(r["applied_norm"] for r in rows)
                        / max(sum(r["proposal_norm"] for r in rows), 1e-12)})
            if "reference_rank" in rows[0]:
                refs = sorted(r["reference_rank"] for r in rows)
                agg["reference_rank_median"] = refs[len(refs) // 2]
        out[proj] = agg
    return out


def main():
    args = parse_args()
    import torch
    import transformers
    from datasets import load_dataset
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    import engram
    from engram.benchmarks import _tofu_official as T

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    timings = {}
    runtime = {
        "git": git_info(repo), "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "engram": f"{engram.__version__} from {engram.__file__}", "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
        # the git commit alone does not identify uncommitted script files
        "sha256": {f: hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), "rb").read()).hexdigest()
                   for f in ("recast_core.py", "recast_tofu.py", "table3_tofu.py", "tofu_capture.py")},
    }
    print(f"[run] args={json.dumps(vars(args))}")
    print(f"[run] runtime={json.dumps(runtime)}")

    import wandb
    run = wandb.init(project=args.wandb_project, name=args.run_name, tags=args.wandb_tags,
                     mode=args.wandb_mode, config={**vars(args), "runtime": runtime})
    print(f"[wandb] mode={args.wandb_mode} url={getattr(run, 'url', None)}")
    results = {"args": vars(args), "runtime": runtime, "timings_s": timings}

    def save_results():
        with open(os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump(to_jsonable(results), f, indent=2)

    # ---------------- data ----------------
    with Stage("load datasets", timings):
        forget = load_dataset(args.dataset_id, args.forget_split)["train"]
        retain = load_dataset(args.dataset_id, args.retain_split)["train"]
        D = {"fp": load_dataset(args.dataset_id, f"{args.forget_split}_perturbed")["train"],
             "ho": load_dataset(args.dataset_id, args.holdout_split)["train"],
             "rp": load_dataset(args.dataset_id, "retain_perturbed")["train"],
             "ra": load_dataset(args.dataset_id, "real_authors_perturbed")["train"],
             "wf": load_dataset(args.dataset_id, "world_facts_perturbed")["train"]}
        print(f"[data] forget={len(forget)} retain={len(retain)} eval={({k: len(v) for k, v in D.items()})}")
    with open(args.anchors_raw) as f:
        anchors = json.load(f)
    ft, rt = anchors["Init Fine."], anchors["Retain"]
    anchor_check = {k: T.evaluate_scores(v, rt, ft) for k, v in (("Init Fine.", ft), ("Retain", rt))}
    print(f"[anchors] {args.anchors_raw}: Init Fine. Overall={anchor_check['Init Fine.']['Overall']:.4f} "
          f"Retain Overall={anchor_check['Retain']['Overall']:.4f}")

    tok = AutoTokenizer.from_pretrained(args.base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print(f"[override] tokenizer pad_token was None -> eos_token {tok.eos_token!r} (as the Table 3 run)")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    model = (AutoModelForCausalLM.from_pretrained(args.base_id, torch_dtype=dtype[args.train_dtype],
                                                  attn_implementation=args.attn_implementation).to(device))
    print(f"[model] {args.base_id} dtype={args.train_dtype} tie_word_embeddings={model.config.tie_word_embeddings}")
    if args.untie_lm_head == "yes":
        untie = untie_output_embedding(model)
        print(f"[untie] lm_head gets its own copy of the input embedding: {json.dumps(untie)}")
        results["untie"] = untie
    head = model.get_output_embeddings().weight
    print(f"[model] lm_head shares the input embedding tensor: "
          f"{head.data_ptr() == model.get_input_embeddings().weight.data_ptr()}")

    # ---------------- phase one ----------------
    targets = select_targets(model, args.targets, blocks=parse_blocks(args.target_blocks))
    restricted = args.variant != "unrestricted"
    head_unrestricted = restricted and args.restricted_head == "unrestricted"
    unrestricted_targets = {"lm_head"} if head_unrestricted else set()
    print(f"[targets] profile={args.targets} blocks={args.target_blocks}: {len(targets)} weights "
          f"({({p: sum(k.endswith(p) for k in targets) for p in PROJECTIONS if any(k.endswith(p) for k in targets)})}); "
          f"outside the restriction: {sorted(unrestricted_targets) or 'none'}")

    def feats(b):
        return {k: b[k].to(device) for k in ("input_ids", "attention_mask", "labels")}

    def gradient_loader(ds):
        return DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=args.grad_batch_size,
                          collate_fn=T.Collator(tok, "right"))

    with Stage("retain gradient", timings):
        grad_retain, n_retain = collect_mean_gradients(model, gradient_loader(retain), targets, feats)
    print(f"[grad] retain: examples={n_retain} global_norm={sum(float(g.square().sum()) for g in grad_retain.values()) ** 0.5:.4e}")
    grad_forget = None
    if restricted:
        restricted_targets = {n: m for n, m in targets.items() if n not in unrestricted_targets}
        with Stage("forget gradient", timings):
            grad_forget, n_forget = collect_mean_gradients(model, gradient_loader(forget), restricted_targets, feats)
        print(f"[grad] forget ({len(restricted_targets)} restricted targets): examples={n_forget} "
              f"global_norm={sum(float(g.square().sum()) for g in grad_forget.values()) ** 0.5:.4e}")

    input_filters, whitening, covs, covs_forget = None, None, None, None
    representation = restricted and args.restriction_representation not in (None, "none")
    contrast = restricted and (args.direction_weighting == "contrast" or args.component_selection == "contrast")
    if restricted and (args.retain_input_filter == "ridge" or args.restriction_whitening == "retain_input" or representation
                       or contrast):
        with Stage("retain input covariance", timings):
            covs, n_tokens = collect_input_covariances(model, gradient_loader(retain), restricted_targets, feats)
        results["retain_input_covariance"] = {"answer_tokens": n_tokens}
    target = args.restriction_forget_output_target if restricted else None
    if representation or target is not None or contrast:
        with Stage("forget input covariance", timings):
            covs_forget, n_tokens_forget = collect_input_covariances(model, gradient_loader(forget), restricted_targets, feats)
        results["forget_input_covariance"] = {"answer_tokens": n_tokens_forget}
    if representation:
        mix = 1.0 if args.representation_mix is None else args.representation_mix
        print(f"[repr] restriction bases from N v = gamma (C_r + eps I) v, N = {1 - mix:g} G_f^T G_f / tr + {mix:g} "
              f"C_f / tr, eps = rho tr(C_r) / d, rho={args.whitening_rho:g}, output side "
              f"{args.restriction_representation}; answer tokens retain {n_tokens}, forget {n_tokens_forget}")
    if restricted and args.restriction_whitening == "retain_input":
        whitening = covs
        print(f"[whiten] restriction bases from G_f (C_r + eps I)^-1/2, eps = rho tr(C_r) / d, "
              f"rho={args.whitening_rho:g}, retain answer tokens={n_tokens}")
    if restricted and args.retain_input_filter == "ridge":
        input_filters, filter_report = {}, {}
        for n, c in covs.items():
            input_filters[n], mu = ridge_input_filter(c, args.retain_filter_kappa)
            # kept weight on average over retain's input energy vs over a uniform direction
            filter_report[n] = {"mu": mu, "kept_retain_weighted": float((input_filters[n] * c).sum() / torch.trace(c)),
                                "kept_uniform": float(torch.trace(input_filters[n]) / c.shape[0])}
        # Positive control: the filter keeps less along the directions retain uses than on average.
        bad = [n for n, r in filter_report.items() if not r["kept_retain_weighted"] < r["kept_uniform"]]
        if bad:
            raise RuntimeError(f"ridge filter does not favour retain directions for {bad[:3]}")
        by_proj = {p: [r for n, r in filter_report.items() if n.endswith(p)] for p in PROJECTIONS}
        print(f"[filter] ridge kappa={args.retain_filter_kappa:g}, retain answer tokens={n_tokens}: " + ", ".join(
            f"{p} kept retain-weighted {sum(r['kept_retain_weighted'] for r in rs) / len(rs):.3f} "
            f"uniform {sum(r['kept_uniform'] for r in rs) / len(rs):.3f}" for p, rs in by_proj.items() if rs))
        results["retain_filter"] = {"answer_tokens": n_tokens, "per_target": filter_report}

    original = {n: m.weight.detach().to("cpu", copy=True) for n, m in targets.items()}  # drift reference

    @torch.no_grad()
    def mean_loss(ds):
        """The loss the source gradients differentiate: the example-weighted mean of each batch's mean NLL."""
        was_training = model.training
        model.eval()
        total, count = 0.0, 0
        for b in gradient_loader(ds):
            x = feats(b)
            size = int(x["input_ids"].shape[0])
            total += float(model(**x).loss) * size
            count += size
        model.train(was_training)
        return total / count

    losses_before = None
    if args.component_diagnostics:
        with Stage("losses before the warm start", timings):
            losses_before = {"forget": mean_loss(forget), "retain": mean_loss(retain)}
    holdout_init = None
    if args.warm_start_scale_target == "holdout":
        with Stage("holdout loss at Init Fine.", timings):
            holdout_init = mean_loss(D["ho"])
        print(f"[scale] target: Init Fine. loss on the holdout split {holdout_init:.4f}")
    full_grads, init_state = None, None
    if args.component_diagnostics and args.epochs > 0 and args.variant != "restricted_semu":
        def full_gradient(ds):
            """Every parameter's mean gradient, defined as the source gradients (eval mode, example-weighted mean
            of batch-mean losses); accumulated on the GPU, returned on the CPU."""
            was_training = model.training
            model.eval()
            params = dict(model.named_parameters())
            flags = {n: t.requires_grad for n, t in params.items()}
            for t in params.values():
                t.requires_grad_(True)
            sums = {n: torch.zeros_like(t, dtype=torch.float32) for n, t in params.items()}
            count = 0
            for b in gradient_loader(ds):
                x = feats(b)
                size = int(x["input_ids"].shape[0])
                model.zero_grad(set_to_none=True)
                model(**x).loss.backward()
                for n, t in params.items():
                    if t.grad is not None:
                        sums[n].add_(t.grad.float(), alpha=float(size))
                count += size
            model.zero_grad(set_to_none=True)
            for n, t in params.items():
                t.requires_grad_(flags[n])
            model.train(was_training)
            return {n: (v / count).cpu() for n, v in sums.items()}

        with Stage("full gradients at Init Fine.", timings):
            full_grads = {"forget": full_gradient(forget), "retain": full_gradient(retain)}
        init_state = {n: t.detach().to("cpu", copy=True) for n, t in model.named_parameters()}
        torch.cuda.empty_cache()
    source = grad_forget
    if restricted and args.restriction_source == "full_data":
        source = {n: mix_sources(grad_forget[n], grad_retain[n], n_forget, n_retain) for n in grad_forget}
        print(f"[source] full_data: (n_f G_f + n_r G_r) / (n_f + n_r) with n_f={n_forget}, n_r={n_retain}")
    with Stage("warm start", timings):
        restrictions, ws_report = warm_start(
            targets, grad_retain, args.s_min, source, args.restriction_evr if source is not None else None,
            unrestricted=unrestricted_targets, alignment_scope=args.restriction_alignment_scope or "full_space",
            layer_scaling=args.shrink_layer_scaling, input_filters=input_filters, whitening=whitening,
            whitening_rho=args.whitening_rho, rank_multiplier=args.restriction_rank_multiplier,
            representation_output=args.restriction_representation if representation else None,
            forget_covariances=covs_forget, retain_covariances=covs if (representation or contrast) else None,
            direction_weighting=args.direction_weighting,
            representation_mix=1.0 if args.representation_mix is None else args.representation_mix,
            forget_output_target=target, component_selection=args.component_selection,
            component_diagnostics=bool(args.component_diagnostics), component_seed=args.seed)
    if args.component_selection or args.component_diagnostics:
        comp = {n: r["components"] for n, r in ws_report.items() if "components" in r}
        arrays = {n: {k: c.pop(k) for k in ("sigma", "contribution") if k in c} for n, c in comp.items()}
        # Positive controls: the fp64 components rebuild the fp32 original update to within fp32 rounding of the
        # proposal (||P(X) - sum D_j|| <= 1e-5 ||X||; the fp32 core of a near-zero update can lose most relative
        # digits), their contributions sum to <G_f, P(X)> up to ||G_f|| times that difference (Cauchy-Schwarz), and
        # the selection kept what it should.
        def control_failed(n, c):
            if c["reconstruction_abs_error"] > 1e-5 * ws_report[n]["proposal_norm"]:
                return True
            gap = abs(c["contribution_sum"] - c["first_order_forget_direct"])
            return gap > c["grad_forget_norm"] * c["reconstruction_abs_error"] + 1e-6 * abs(c["first_order_forget_direct"]) + 1e-12

        bad = [n for n, c in comp.items() if control_failed(n, c)]
        if not comp or bad:
            raise RuntimeError(f"component decomposition control failed at {bad[:3]} ({len(comp)} targets): " + "; ".join(
                f"{n}: reconstruction {comp[n]['reconstruction_abs_error']:.3e} vs proposal "
                f"{ws_report[n]['proposal_norm']:.3e}, sum c_j {comp[n]['contribution_sum']:.6e} vs "
                f"{comp[n]['first_order_forget_direct']:.6e}" for n in bad[:3]))
        worst = max(comp, key=lambda n: comp[n]["reconstruction_rel_error"])
        if args.component_selection in ("signed", "random") and any(c["kept"] != c["positive"] for c in comp.values()):
            raise RuntimeError("the selection kept a different number of components than have c_j > 0")
        if args.component_selection in (None, "full", "uniform_matched") and any(
                c["kept"] != c["components"] and not (c["kept"] == 0 and c["positive"] == 0) for c in comp.values()):
            raise RuntimeError("a full or uniform update dropped components")  # gamma = 0 only without c_j > 0

        def summed(rows):
            keys = ("components", "positive", "negative", "kept", "energy", "contribution_sum", "contribution_positive",
                    "contribution_negative", "predicted_forget_change", "predicted_retain_change")
            out = {k: sum(r[k] for r in rows) for k in keys}
            energy = sum(r["energy"] for r in rows)
            for k in ("energy_positive_fraction", "energy_negative_fraction", "applied_energy_fraction"):
                out[k] = sum(r[k] * r["energy"] for r in rows) / energy if energy > 0 else 0.0
            out["original_norm"] = sum(r["original_norm"] ** 2 for r in rows) ** 0.5
            out["applied_norm"] = sum(r["applied_norm"] ** 2 for r in rows) ** 0.5
            out["effective_rank_original_mean"] = sum(r["effective_rank_original"] for r in rows) / len(rows)
            out["effective_rank_applied_mean"] = sum(r["effective_rank_applied"] for r in rows) / len(rows)
            out["max_reconstruction_rel_error"] = max(r["reconstruction_rel_error"] for r in rows)
            out["targets"] = len(rows)
            return out

        comp_summary = {p_: summed([c for n, c in comp.items() if n.endswith(p_)]) for p_ in PROJECTIONS
                        if any(n.endswith(p_) for n in comp)}
        comp_summary["all"] = summed(list(comp.values()))
        for p_, c in comp_summary.items():
            print(f"[components] {p_:9s} selection={args.component_selection or 'none'}: components {c['components']} "
                  f"c_j>0 {c['positive']} kept {c['kept']}; energy with c_j>0 {c['energy_positive_fraction']:.4f}, "
                  f"c_j<0 {c['energy_negative_fraction']:.4f}; applied energy {c['applied_energy_fraction']:.4f}, "
                  f"norm {c['applied_norm']:.4e} / {c['original_norm']:.4e}; effective rank "
                  f"{c['effective_rank_applied_mean']:.1f} / {c['effective_rank_original_mean']:.1f}; sum c_j "
                  f"{c['contribution_sum']:.4e} (+{c['contribution_positive']:.4e} {c['contribution_negative']:.4e}); "
                  f"predicted dL_f {c['predicted_forget_change']:.4e} dL_r {c['predicted_retain_change']:.4e}")
        print(f"[control] components: rebuild the original update (max rel. error "
              f"{comp_summary['all']['max_reconstruction_rel_error']:.2e} at {worst}, abs "
              f"{comp[worst]['reconstruction_abs_error']:.2e}, update norm {comp[worst]['original_norm']:.2e}), "
              f"sum c_j = <G_f, P(X)>, selection counts")
        results["components"] = {"selection": args.component_selection, "per_projection": comp_summary}
        if args.component_diagnostics:
            with open(os.path.join(args.out_dir, "components.json"), "w") as f:
                json.dump(to_jsonable({n: {**comp[n], **arrays[n]} for n in comp}), f)
            with Stage("losses after the warm start", timings):
                losses_after = {"forget": mean_loss(forget), "retain": mean_loss(retain)}
            check = {}
            for side, key in (("forget", "predicted_forget_change"), ("retain", "predicted_retain_change")):
                actual = losses_after[side] - losses_before[side]
                predicted = comp_summary["all"][key]
                check[side] = {"before": losses_before[side], "after": losses_after[side], "actual_change": actual,
                               "predicted_change": predicted, "abs_error": abs(actual - predicted),
                               "rel_error": abs(actual - predicted) / abs(predicted) if abs(predicted) > 1e-8 else None}
                print(f"[first-order] {side}: loss {losses_before[side]:.4f} -> {losses_after[side]:.4f}, actual change "
                      f"{actual:.4e}, predicted <G, Delta W> {predicted:.4e}, abs error {check[side]['abs_error']:.4e}, "
                      f"rel error {'-' if check[side]['rel_error'] is None else format(check[side]['rel_error'], '.3g')}")
            results["components"]["first_order_check"] = check
    if target is not None:
        # Positive control: every active target reaches the forget-output target, or uses its whole family.
        reached = {n: r for n, r in ws_report.items() if "forget_output_change" in r}
        short = [n for n, r in reached.items() if r["forget_output_change"] < target and r["rank"] < r["total_rank"]]
        if not reached or short:
            raise RuntimeError(f"forget-output budget missed the target {target:g} at {short[:3]} "
                               f"({len(reached)} targets reported)")
        full = [n for n, r in reached.items() if r["forget_output_change"] < target]
        print(f"[budget] forget-output target {target:g}, forget answer tokens {n_tokens_forget}: " + ", ".join(
            f"{p} q_f {sum(r['forget_output_change'] for r in rs) / len(rs):.4f} rank "
            f"{sum(r['rank'] for r in rs) / len(rs):.0f} (EVR rank {sum(r['reference_rank'] for r in rs) / len(rs):.0f})"
            for p in PROJECTIONS for rs in [[r for n, r in reached.items() if n.endswith(p)]] if rs)
            + f"; at full rank below the target: {len(full)}")
        results["forget_output_budget"] = {"target": target, "below_target_at_full_rank": full}
    if representation:
        # Positive control: the representation covectors have a higher forget / retain input-energy ratio than
        # the plain covectors of the same rank.
        def contrast(vh, cf, cr):
            vh = vh.to(device=cf.device, dtype=cf.dtype)
            return float(torch.trace(vh @ cf @ vh.transpose(0, 1)) / torch.trace(vh @ cr @ vh.transpose(0, 1)))
        ratio = {}
        for n, r in restrictions.items():
            if r.basis.active:
                plain = svd_basis(source[n], args.restriction_evr, rank=r.basis.rank)
                ratio[n] = {"representation": contrast(r.basis.vh, covs_forget[n], covs[n]),
                            "plain": contrast(plain.vh, covs_forget[n], covs[n]),
                            "kept_forget_gradient_energy": r.basis.evr, "plain_kept_forget_gradient_energy": plain.evr}
        d_mean = sum(e["representation"] for e in ratio.values()) / len(ratio)
        p_mean = sum(e["plain"] for e in ratio.values()) / len(ratio)
        if not d_mean > p_mean:
            raise RuntimeError(f"representation bases show no higher forget/retain contrast than plain ones "
                               f"({d_mean} vs {p_mean})")
        print(f"[repr] forget/retain input energy ratio per basis: representation {d_mean:.3f}, plain {p_mean:.3f}; "
              f"forget-gradient energy kept: representation "
              f"{sum(e['kept_forget_gradient_energy'] for e in ratio.values()) / len(ratio):.3f}, plain "
              f"{sum(e['plain_kept_forget_gradient_energy'] for e in ratio.values()) / len(ratio):.3f}")
        results["representation"] = {"rho": args.whitening_rho, "output_side": args.restriction_representation,
                                     "input_mix": mix, "per_target": ratio}
    if whitening is not None:
        # Positive control: the whitened covectors carry less retain input energy than the plain ones.
        def relative_energy(vh, c):  # mean retain energy per covector / mean energy of a direction
            vh = vh.to(device=c.device, dtype=c.dtype)
            return float(torch.trace(vh @ c @ vh.transpose(0, 1)) / vh.shape[0] / (torch.trace(c) / c.shape[0]))
        energy = {}
        for n, r in restrictions.items():
            if r.basis.active:
                # with an explicit budget the plain basis is compared at the same rank
                plain = svd_basis(source[n], args.restriction_evr, rank=r.basis.rank if (
                    args.restriction_rank_multiplier is not None or target is not None) else None)
                energy[n] = {"whitened": relative_energy(r.basis.vh, whitening[n]),
                             "plain": relative_energy(plain.vh, whitening[n]),
                             "rank_whitened": r.basis.rank, "rank_plain": plain.rank}
        w_mean = sum(e["whitened"] for e in energy.values()) / len(energy)
        p_mean = sum(e["plain"] for e in energy.values()) / len(energy)
        if not w_mean < p_mean:
            raise RuntimeError(f"whitened bases carry no less retain energy than plain ones ({w_mean} vs {p_mean})")
        print(f"[whiten] retain input energy per covector / average direction: whitened {w_mean:.3f}, plain "
              f"{p_mean:.3f}; " + ", ".join(
                  f"{p} rank whitened/plain {sum(e['rank_whitened'] for e in es) / len(es):.0f}/"
                  f"{sum(e['rank_plain'] for e in es) / len(es):.0f}"
                  for p in PROJECTIONS for es in [[e for k, e in energy.items() if k.endswith(p)]] if es))
        results["whitening"] = {"rho": args.whitening_rho, "per_target": energy}
    by_type = parse_type_scales(args.warm_start_scale_by_type) if args.warm_start_scale_by_type else None
    if args.warm_start_scale is not None or args.warm_start_scale_target is not None or by_type:
        # W(t) = W_Init + t (W* - W_Init), each target rebuilt from its Init weights for every t
        starred = {n: m.weight.detach().clone() for n, m in targets.items()}
        init_w = {n: original[n].to(device) for n in targets}

        def t_of(n, t):  # the target's own t under --warm_start_scale_by_type
            return by_type[n.rsplit(".", 1)[-1]] if by_type else t

        def set_scale(t):
            with torch.no_grad():
                for n, m in targets.items():
                    m.weight.copy_(scale_update(init_w[n], starred[n], t_of(n, t)).to(m.weight.dtype))

        search = []
        scale = args.warm_start_scale
        if args.warm_start_scale_target == "holdout":
            with Stage("scale search", timings):
                lo, hi = 0.0, 1.0 / (1.0 - args.s_min)
                set_scale(hi)
                f_hi = mean_loss(forget)
                search.append((hi, f_hi))
                if f_hi <= holdout_init:
                    scale = hi  # even s = 0 does not reach the target
                else:
                    for _ in range(16):  # forget loss grows with t; bracket [lo, hi] to ~2e-5 in t
                        mid = (lo + hi) / 2
                        set_scale(mid)
                        f_mid = mean_loss(forget)
                        search.append((mid, f_mid))
                        lo, hi = (mid, hi) if f_mid < holdout_init else (lo, mid)
                    scale = hi
        set_scale(scale)
        # Positive control: every target moved by exactly t times its warm-start update.
        moved = {n: float((m.weight.detach().float() - init_w[n].float()).norm()) for n, m in targets.items()}
        full = {n: float((starred[n].float() - init_w[n].float()).norm()) for n in targets}
        bad = [n for n in targets if abs(moved[n] - t_of(n, scale) * full[n]) > 1e-4 * max(full[n], 1e-12) + 1e-6]
        if bad:
            raise RuntimeError(f"scaled update is not t times the warm start at {bad[:3]}")
        for n in restrictions:
            restrictions[n] = type(restrictions[n])(baseline=targets[n].weight.detach().clone(),
                                                    basis=restrictions[n].basis)
        scale_report = {"t": scale, "displacement": sum(v * v for v in moved.values()) ** 0.5,
                        "displacement_t1": sum(v * v for v in full.values()) ** 0.5, "search": search}
        if by_type:
            scale_report.update({"t_by_type": by_type, "s_equivalent_by_type": {
                k: 1.0 - (1.0 - args.s_min) * v for k, v in by_type.items()}, "displacement_by_type": {
                k: sum(v * v for n, v in moved.items() if n.endswith(k)) ** 0.5 for k in by_type}})
        else:
            scale_report["s_equivalent"] = 1.0 - (1.0 - args.s_min) * scale
        if holdout_init is not None:
            scale_report.update({"target_holdout_loss": holdout_init, "forget_loss": mean_loss(forget)})
        print((f"[scale] t per type {by_type} (floor equivalent s {scale_report['s_equivalent_by_type']}, displacement "
               f"{({k: round(v, 3) for k, v in scale_report['displacement_by_type'].items()})})" if by_type else
               f"[scale] t = {scale:.5f} (uniform floor equivalent s = {scale_report['s_equivalent']:.4f})")
              + f"; displacement {scale_report['displacement']:.3f} of {scale_report['displacement_t1']:.3f} at t = 1"
              + (f"; forget loss {scale_report['forget_loss']:.4f} for the holdout target {holdout_init:.4f} after "
                 f"{len(search)} evaluations" if holdout_init is not None else ""))
        print("[control] scaled update: every target moved by its t times its warm-start update")
        results["warm_start_scale"] = scale_report
        del starred, init_w
    if args.variant == "unrestricted" and args.s_min == 1.0:
        # Positive control: s = max(cos, 1) = 1 for every row, so phase one leaves the start model as it is.
        moved = [n for n, r in ws_report.items() if r["weight_rel_change"] != 0.0]
        if moved:
            raise RuntimeError(f"s_min = 1 changed {len(moved)} targets, e.g. {moved[:3]}")
        print("[control] s_min = 1: phase one left every target unchanged; phase two starts from --base_id")
    del grad_retain, grad_forget, source, input_filters, whitening, covs, covs_forget
    torch.cuda.empty_cache()
    ws_summary = summarize_by_projection(ws_report)
    for proj, agg in ws_summary.items():
        print(f"[warm-start] {proj}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()})}")
    results["warm_start"] = {"per_projection": ws_summary, "per_target": ws_report}

    # ---------------- restriction backend and trainable set ----------------
    if args.variant == "restricted_semu":
        adapter_report = apply_semu_adapters(model, restrictions)
        print(f"[adapter] {json.dumps(adapter_report)}")
        results["adapter"] = adapter_report
        for p in model.parameters():
            p.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, SemuAdapterLinear):
                m.weight.requires_grad_(True)
        for name in unrestricted_targets:  # shrunk without projection, trained freely (fc rule)
            dict(model.named_modules())[name].weight.requires_grad_(True)
        free = free_parameters(model, args.phase2_free)
        for params in free.values():
            for p in params.values():
                p.requires_grad_(True)
        free_summary = {g: {"tensors": len(ps), "parameters": sum(p.numel() for p in ps.values())} for g, ps in free.items()}
        print(f"[train] trained without an adapter (--phase2_free {args.phase2_free}): {json.dumps(free_summary)}")
        results["phase2_free"] = free_summary
    else:  # unrestricted, restricted_shrink: native weights, every parameter trains
        free = {}
        for p in model.parameters():
            p.requires_grad_(True)
        if args.freeze_input_embedding == "yes":
            model.get_input_embeddings().weight.requires_grad_(False)
        if args.variant == "restricted_shrink":
            print("[train] restricted_shrink: warm start restricted, no adapter; phase two trains every parameter")
    # The head outside the phase-one targets (Classification --no-recast_shrink_fc): trained freely or frozen.
    model.get_output_embeddings().weight.requires_grad_(
        args.train_lm_head == "yes" or "lm_head" in unrestricted_targets or args.variant != "restricted_semu"
        and "lm_head" in targets)
    if args.train_lm_head == "no" and "lm_head" in targets:
        raise ValueError("--train_lm_head no conflicts with a head that is a phase-one target")
    if model.get_input_embeddings().weight.requires_grad and args.untie_lm_head == "yes":
        raise ValueError("an untied head with a trainable input embedding is not a configuration of this campaign")
    print(f"[train] input embedding trainable: {model.get_input_embeddings().weight.requires_grad}; "
          f"lm_head trainable: {model.get_output_embeddings().weight.requires_grad}; shared tensor: "
          f"{model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable parameters {n_train:,} / {n_all:,} ({100.0 * n_train / n_all:.3f}%)")

    # ---------------- phase two: retain NLL only ----------------
    change = {}
    if args.epochs == 0:
        print("[train] --epochs 0: no phase two; the warm-started model is evaluated as it is")
        results["phase_two"] = {"skipped": True}
    else:
        retain_qa = T.QADataset(retain, tok, akey="answer")

        class AnchoredRetain(Dataset):
            """OpenUnlearning ForgetRetainDataset(anchor='forget'), retain side only: len(forget) items,
            each a uniformly random retain example (torch.randint, seeded by the Trainer)."""

            def __len__(self):
                return len(forget)

            def __getitem__(self, idx):
                return retain_qa[int(torch.randint(0, len(retain_qa), (1,)).item())]

        train_ds = AnchoredRetain() if args.epoch_length == "forget" else retain_qa
        collator = T.Collator(tok, "right")

        def collate(items):
            batch = collator(items)
            batch.pop("index")
            return batch

        effective_batch = args.per_device_batch_size * args.grad_accum * max(1, torch.cuda.device_count())
        piecewise = args.lr_scheduler == "piecewise_epochs"
        lr_points = parse_points(args.lr_points) if piecewise else None
        # OpenUnlearning formula for Hugging Face schedulers; the piecewise schedule carries its own warmup.
        warmup_steps = 0 if piecewise else int(args.warmup_epochs * len(train_ds) // effective_batch)

        class PiecewiseTrainer(Trainer):
            """Learning rate linear through (epoch, lr) points; replaces the Hugging Face scheduler."""

            def create_scheduler(self, num_training_steps, optimizer=None):
                if self.lr_scheduler is None:
                    steps_per_epoch = num_training_steps / float(self.args.num_train_epochs)
                    self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer if optimizer is not None else self.optimizer,
                        lambda step: piecewise_linear(lr_points, step / steps_per_epoch) / args.lr)
                    self._created_lr_scheduler = True
                    print(f"[override] piecewise schedule replaces lr_scheduler_type=constant: points={lr_points}, "
                          f"steps/epoch={steps_per_epoch:g}, total steps={num_training_steps}")
                return self.lr_scheduler

        targs = TrainingArguments(
            output_dir=os.path.join(args.out_dir, "trainer"), per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, weight_decay=args.weight_decay,
            num_train_epochs=args.epochs, warmup_steps=warmup_steps,
            lr_scheduler_type="constant" if piecewise else args.lr_scheduler,
            optim=args.optim, max_grad_norm=args.max_grad_norm, bf16=True, logging_steps=5, save_strategy="no",
            report_to=[], seed=args.seed, remove_unused_columns=False, gradient_checkpointing=False,
        )
        teacher, teacher_checksums = None, None
        if "kl" in args.retain_loss:  # the model before phase one, frozen
            teacher = (AutoModelForCausalLM.from_pretrained(args.base_id, torch_dtype=dtype[args.train_dtype],
                                                            attn_implementation=args.attn_implementation)
                       .to(device).eval())
            teacher.requires_grad_(False)
            teacher_checksums = frozen_checksums(teacher)
            print(f"[train] retain_loss={args.retain_loss} kl_weight={args.kl_weight}: teacher {args.base_id} (Init Fine.)")

        class RetainLoss:
            """NLL from the Trainer; with a teacher, KL(teacher || model) averaged over non-padding positions."""
            kl_log = []

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                nll, outputs = super().compute_loss(model, inputs, return_outputs=True,
                                                    num_items_in_batch=num_items_in_batch)
                if teacher is None:
                    return (nll, outputs) if return_outputs else nll
                with torch.no_grad():
                    ref = teacher(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
                mask = inputs["attention_mask"].bool()
                kl = F.kl_div(F.log_softmax(outputs.logits[mask].float(), dim=-1),
                              F.log_softmax(ref[mask].float(), dim=-1), reduction="batchmean", log_target=True)
                self.kl_log.append(float(kl.detach()))
                # The Trainer leaves the loss unscaled when the model normalises the NLL over the whole
                # accumulation window (num_items_in_batch); the per-micro-batch KL mean then needs the 1/GA.
                if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
                    kl = kl / self.current_gradient_accumulation_steps
                loss = args.kl_weight * kl + (nll if "nll" in args.retain_loss else 0.0)
                return (loss, outputs) if return_outputs else loss

        trajectory = []
        # the evaluation sets, all small: retain = retain_perturbed (the 400 retain items of Model Utility)
        tracked = {"forget": forget, "retain": D["rp"], "real_authors": D["ra"], "world_facts": D["wf"]}

        @torch.no_grad()
        def trajectory_point(epoch, log_history=()):
            """Answer-token NLL and exact memorization (teacher-forced, as metric_em, in the training dtype) per
            set, and the epoch's mean training loss; every random state is restored (DataLoader iterators draw a
            seed)."""
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate(), np.random.get_state())
            was_training = model.training
            model.eval()
            losses_ = [h["loss"] for h in log_history if "loss" in h and epoch - 1 < h["epoch"] <= epoch]
            point = {"epoch": epoch, "train_loss_mean": sum(losses_) / len(losses_) if losses_ else None}
            with Stage(f"trajectory epoch {epoch:g}", timings):
                for name, ds in tracked.items():
                    nll, count, em = 0.0, 0, []
                    for b in DataLoader(T.QADataset(ds, tok, akey="answer"), batch_size=args.eval_batch_size,
                                        collate_fn=collator):
                        x = feats(b)
                        logits = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).logits[:, :-1]
                        labels = x["labels"][:, 1:]
                        mask = labels != -100
                        nll += float(F.cross_entropy(logits[mask].float(), labels[mask], reduction="sum"))
                        count += int(mask.sum())
                        # per item: the fraction of answer tokens the argmax predicts (metric_em), then the mean
                        hit = ((logits.argmax(-1) == labels) & mask).sum(1)
                        em += [float(h) / int(n) for h, n in zip(hit, mask.sum(1)) if int(n) > 0]
                    point[name] = {"nll": nll / count, "em": sum(em) / len(em)}
            model.train(was_training)
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state_all(rng[1])
            random.setstate(rng[2])
            np.random.set_state(rng[3])
            trajectory.append(point)
            results["trajectory"] = trajectory
            print(f"[trajectory] epoch {epoch:g}: train loss "
                  f"{'-' if point['train_loss_mean'] is None else format(point['train_loss_mean'], '.4f')}; "
                  + "; ".join(f"{n} nll {point[n]['nll']:.4f} em {point[n]['em']:.4f}" for n in tracked))
            run.log({"trajectory/epoch": epoch, **{f"trajectory/{n}_{k}": v for n in tracked for k, v in point[n].items()}})

        class Trajectory(TrainerCallback):
            def on_epoch_end(self, train_args, state, control, **kwargs):
                trajectory_point(round(state.epoch), state.log_history)

        trainer_class = type("RetainTrainer", (RetainLoss, PiecewiseTrainer if piecewise else Trainer), {})
        trainer = trainer_class(model=model, args=targs, train_dataset=train_ds, data_collator=collate,
                                callbacks=[Trajectory()] if args.eval_trajectory else None)
        print(f"[train] epoch_length={args.epoch_length} items/epoch={len(train_ds)} effective_batch={effective_batch} "
              f"warmup_steps={warmup_steps} optimizer={targs.optim} scheduler={args.lr_scheduler} "
              f"max_grad_norm={targs.max_grad_norm} ({'clipping off' if not targs.max_grad_norm else 'clipping on'})")
        snapshot, checksums = snapshot_trainable(model), frozen_checksums(model)
        if args.eval_trajectory:
            trajectory_point(0)
        with Stage("phase two", timings):
            trainer.train()
        if args.eval_trajectory and [p_["epoch"] for p_ in trajectory] != list(range(int(args.epochs) + 1)):
            raise RuntimeError(f"trajectory epochs {[p_['epoch'] for p_ in trajectory]}, expected 0..{args.epochs:g}")
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        change = change_report(model, snapshot, checksums)
        post_warm_start = {n: (restrictions[n].baseline if n in restrictions else snapshot[n + ".weight"]) for n in targets}
        groups = [g for g in PROJECTIONS if any(n.endswith(g) for n in targets)]
        drift = {"vs_original": drift_report(model, original, groups),
                 "vs_post_warm_start": drift_report(model, post_warm_start, groups)}
        for ref, rep in drift.items():
            print(f"[drift] {ref}: " + ", ".join(f"{g} norm_ratio={v['norm_ratio']:.4f} cos={v['cosine']:.4f}" for g, v in rep.items()))
        # Positive control for --phase2_free: every freed group must have moved.
        free_change = {g: {k: change_report(model, {n: snapshot[n] for n in ps}, {})[k]
                           for k in ("trainable_entries", "trainable_entries_changed_fraction", "trainable_rel_change")}
                       for g, ps in free.items()}
        if free_change:
            print(f"[train] free-group changes: {json.dumps(free_change)}")
        del snapshot, original, post_warm_start
        print(f"[train] optimizer steps={trainer.state.global_step} loss first={losses[0] if losses else None} "
              f"last={losses[-1] if losses else None}")
        # Positive control for the schedule: the learning rate the optimizer actually logged.
        lr_log = [(h["step"], h["learning_rate"]) for h in trainer.state.log_history if "learning_rate" in h]
        if lr_log:
            spe = trainer.state.global_step / args.epochs
            picks = [min(lr_log, key=lambda e: abs(e[0] - x * spe)) for x in (0.5, 1, 3, 5, 7.5, args.epochs)]
            print("[train] logged lr at epoch 0.5/1/3/5/7.5/end: " + ", ".join(f"step {s}: {v:.3g}" for s, v in picks))
        print(f"[train] changes: {json.dumps(change)}")
        kl_log = trainer.kl_log
        if teacher is not None:
            # Positive controls: the KL term was computed on every micro-batch, and the teacher never moved.
            expected = math.ceil(len(train_ds) / args.per_device_batch_size) * int(args.epochs)
            if args.epochs == int(args.epochs) and len(kl_log) != expected:
                raise RuntimeError(f"KL computed on {len(kl_log)} micro-batches, expected {expected}")
            if frozen_checksums(teacher) != teacher_checksums:
                raise RuntimeError("the KL teacher changed during phase two")
            k = max(1, len(kl_log) // 20)
            print(f"[train] KL(Init Fine. || model) per position: micro-batches={len(kl_log)} "
                  f"first {k} mean={sum(kl_log[:k]) / k:.4f} last {k} mean={sum(kl_log[-k:]) / k:.4f}; teacher unchanged")
            del teacher
        results["phase_two"] = {"steps": trainer.state.global_step, "losses": losses, "log_history": trainer.state.log_history,
                                "changes": change, "drift": drift,
                                "free_groups": free_change, "warmup_steps": warmup_steps, "items_per_epoch": len(train_ds),
                                "kl_per_microbatch": kl_log}
        del trainer
        torch.cuda.empty_cache()
    if args.save_model_dir:
        with Stage("save model", timings):
            model.save_pretrained(args.save_model_dir)
            tok.save_pretrained(args.save_model_dir)
        print(f"[save] model ({args.train_dtype}) and tokenizer -> {args.save_model_dir}")
        results["saved_model"] = args.save_model_dir
    save_results()

    if full_grads is not None:
        with Stage("losses after phase two", timings):
            after = {"forget": mean_loss(forget), "retain": mean_loss(retain)}
        params = dict(model.named_parameters())
        update = {}
        for scope, names in (("targets", sorted(n + ".weight" for n in targets)), ("all", sorted(init_state))):
            sq, dot_f, dot_r = 0.0, 0.0, 0.0
            for n in names:
                d = params[n].detach().to("cpu", torch.float64) - init_state[n].double()
                sq += float(d.square().sum())
                dot_f += float((full_grads["forget"][n].double() * d).sum())
                dot_r += float((full_grads["retain"][n].double() * d).sum())
            update[scope] = {"tensors": len(names), "update_norm": sq ** 0.5, "predicted_forget_change": dot_f,
                             "predicted_retain_change": dot_r}
        actual = {side: after[side] - losses_before[side] for side in after}
        print(f"[phase-two check] loss forget {losses_before['forget']:.4f} -> {after['forget']:.4f} ({actual['forget']:+.4f}), "
              f"retain {losses_before['retain']:.4f} -> {after['retain']:.4f} ({actual['retain']:+.4f}); " + "; ".join(
                  f"{scope} ({u['tensors']} tensors): ||theta - theta_Init|| {u['update_norm']:.3f}, predicted forget "
                  f"{u['predicted_forget_change']:+.4f} retain {u['predicted_retain_change']:+.4f}"
                  for scope, u in update.items()))
        results["components"]["phase_two_check"] = {"loss_init": losses_before, "loss_after": after,
                                                     "actual_change": actual, "update": update}
        del full_grads, init_state, params
        save_results()

    # ---------------- evaluation ----------------
    model.eval()
    if args.eval_dtype != args.train_dtype:
        model.to(dtype[args.eval_dtype])
        print(f"[eval] cast {args.train_dtype} -> {args.eval_dtype} for evaluation (anchors were scored in bfloat16)")
    row = f"ReCAST {args.variant} (s_min={args.s_min:g}){' warm start only' if args.epochs == 0 else ''}"
    # compute_full_tofu discards the nine Model Utility sub-metrics and the generations; keep them.
    with TofuCapture(T, ("forget", "retain", "real_authors", "world_facts")) as capture:
        with Stage(f"evaluate {row}", timings):
            raw = T.compute_full_tofu(model, tok, D, bs=args.eval_batch_size)
    subs, generations = capture.check(raw["model_utility"], forget_rouge=raw["forget_Q_A_ROUGE"])
    print("[control] capture: sub-metrics give the reported Model Utility; per-split ROUGE means match")
    raw["_model_utility_subs"] = subs
    print(f"[result] model utility sub-metrics: {json.dumps({k: round(v, 4) for k, v in subs.items()})}")
    with open(os.path.join(args.out_dir, "raw_metrics.json"), "w") as f:
        json.dump(to_jsonable({row: raw}), f)
    with open(os.path.join(args.out_dir, "generations.json"), "w") as f:
        json.dump(generations, f)
    scores = T.evaluate_scores(raw, rt, ft)
    results.update({"row": row, "scores": scores, "anchor_check": anchor_check})
    save_results()
    print(f"[result] {row}: " + ", ".join(f"{c}={scores[c]:.4f}" for c in COLUMNS))
    for ref in ("Engram (alpha=0.6)", "Engram (alpha_W-Norm)"):
        print(f"[result] paper {ref}: " + ", ".join(f"{c}={PAPER_TABLE3[ref][c]:.4f}" for c in COLUMNS))

    run.log({**{f"scores/{c}": scores[c] for c in COLUMNS}, **{f"phase_two/{k}": v for k, v in change.items()
                                                               if isinstance(v, (int, float))}})
    results["cuda_max_memory_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    save_results()
    run.summary.update({"timings_s": timings, "warm_start": to_jsonable(ws_summary), "row": row,
                        "cuda_max_memory_gb": results["cuda_max_memory_gb"],
                        **({"components": to_jsonable(results["components"])} if "components" in results else {})})
    run.finish()


if __name__ == "__main__":
    main()
