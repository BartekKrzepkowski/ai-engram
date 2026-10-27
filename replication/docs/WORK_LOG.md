# Work Log

Rules for this file are in `AGENTS.md` (Git, Documentation, and Reporting).
`grep -n '^- \[ \]'` lists open work; `grep -n 'Deviation:'` lists every
deviation from the paper or upstream code.

## 2026-10-21

- [x] 22:20 Goal (user): "naszym celem tutaj jest zreplikowanie podstawowego
  badania z ich papieru, a konkretnie table 3", "chodziło mi o eksperyment z
  LLMami" -- Table 3 of arXiv 2606.14997 (Llama-3.2-1B-Instruct, TOFU
  `forget10`, Appendix E.1-E.3), on branch `replication/engram-tofu-table3`
  from upstream `main` 17f1a54 (ai-engram 0.11.0). Campaign type
  `method-faithful`, tag `engram_tofu_table3_faithful`.
- [x] 22:20 Decision (user): scope "Engram + kotwice (Recommended)" -- the
  rows Init Fine., Retain, Engram (alpha = 0.6) and Engram (alpha_W-Norm) are
  replicated with one evaluation code; the eight trained baselines are quoted
  from the paper and marked as not replicated.
- [x] 22:20 Agent policy ported from `reload-unlearning` (user: "przejrzyj te
  pliki i zdecyduj co przenieść do tego repozytorium"): `CLAUDE.md` (as is),
  `AGENTS.md` (project sections rewritten: scope, upstream fidelity, layout,
  scientific invariants; RELOAD, ECO, contextual fine-tuning and Anthropic API
  rules dropped), `replication/docs/agent-guides/{CLUSTER_ENVIRONMENT,SLURM_AND_WANDB,WORKFLOW,RULE_RATIONALE}.md`
  (paths, environment, W&B project and tags adapted). Not ported: the
  `reload-unlearning` work log (its history is about RELOAD) and all
  `Reload-*` code. Guides live under `replication/docs/` because upstream
  `docs/` is the MkDocs site. `replication/logs` is a symlink to
  `archive/plgkrzepk/ai-engram_logs`, registered in `.git/info/exclude`.
  Documentation only, no gate.
- [x] 22:27 Decision (user): "tak, zgadzam sie na 1-3" -- accepts the environment
  spec, the build job and the Table 3 downloads (entries 22:20).
- [x] 22:20 Environment `ai-engram-gh200`, spec accepted (22:27):
  `replication/environment/ai_engram_gh200.yaml` (Python 3.11, the upstream
  CI version), `requirements_gh200.txt` (`torch==2.6.0` cu126, as in
  `reload-llm-gh200` and verified upstream in `CHANGELOG.md` 0.8.0;
  `transformers==4.57.6`, the latest 4.x; `datasets==3.6.0`, `scipy==1.15.3`,
  `rouge-score==0.1.2`, `numpy==2.2.6`, `scikit-learn==1.7.2`,
  `pytest==8.4.2`, `wandb==0.26.1`), ai-engram installed `--no-deps -e`.
  Upstream pins nothing, so the pins are a choice, not a deviation. Upstream
  gap: the `tofu` extra omits `scikit-learn`, which
  `engram.benchmarks._tofu_official` imports; the spec adds it. Launcher
  `replication/scripts/bash/environment/build_env_gh200.sh` (30 min limit;
  `reload-llm-gh200` built in 7:33 plus a 2:14 repair) ends with a CUDA
  smoke test, the TOFU import path and the upstream CPU test suite offline.
  Gate: `bash -n`. Build job 24947042 (1 GH200, 8 CPU, 32 GB, 30 min limit:
  estimate ~10 min, new cp311 wheels plus an upstream test suite of unknown
  duration), log `replication/logs/slurm/ai-engram-env-24947042.out`.
  Result 24947042 (22:29): FAILED after 1:17. Conda created the prefix (Python
  3.11.17, pip 26.2.2), then `pip install torch==2.6.0` from the cu126 index found
  "versions: none", with no retry or fetch warning in the log. The wheel
  `torch-2.6.0+cu126-cp311-cp311-linux_aarch64.whl` exists, and pip 26.2.2 in an x86
  scratch venv with `--platform linux_aarch64` lists `2.6.0+cu126` for Python 3.10
  and 3.11, so neither pip nor Python explains it; no pip config or `PIP_*` other
  than `PIP_CACHE_DIR`. Cause unknown. The launcher now prints the pip version and
  `pip index versions torch` before installing. Resubmitted in repair mode
  (`AI_ENGRAM_ENV_REPAIR=1`, existing prefix) as 24947364, same limit.
  Result 24947364 (22:56): COMPLETED in 22:11 on x1002c0s0b1n0. pip now listed
  `2.6.0+cu126` (the first failure was transient); the torch wheel download was
  interrupted three times and resumed. Installed `torch 2.6.0+cu126`, all pins as
  specified, `huggingface-hub 0.36.2`, `ai-engram 0.11.0` editable from the
  checkout; `pip check` clean; freeze in `<prefix>/ai-engram-gh200.freeze.txt`.
  Gates: CUDA 12.6 bf16 matmul on `NVIDIA GH200 120GB`; `engram`,
  `engram.benchmarks.tofu` and `_tofu_official` import; upstream tests offline
  112 passed, 4 skipped (two MoE tests that need transformers 5.x, the two gated TOFU
  tests) in 13:00, which took most of the job time.
- [x] 22:27 Downloads for Table 3: job 24947365 (24947043 cancelled while pending,
  its dependency on the failed build never satisfied)
  (`replication/scripts/bash/download/prefetch_table3_gh200.sh`, new
  `replication/scripts/prefetch_hf.py`; `--dependency=afterok:24947364`, 1 GH200,
  15 min limit: ~5.2 GB, 20 s-2 min at the 42-270 MB/s of earlier jobs):
  `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` and `_retain90` (2.47 GB
  safetensors each), `madhurjindal/autonlp-Gibberish-Detector-492513457`
  (safetensors, `onnx/` skipped), and the `locuslab/TOFU` configs
  `forget10_perturbed holdout10 retain_perturbed real_authors_perturbed
  world_facts_perturbed full`; all public, not gated (HF API). Revisions to
  `runtime_artifacts/ai-engram/hf_manifest_table3_24947365.json`. Positive
  control: a second pass with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` loads
  every config, tokenizer and dataset config from the cache. Gate:
  `py_compile` and `bash -n`.
  Result 24947365 (22:58): COMPLETED in 1:53. Revisions: `_full` 88e31200 and
  `_retain90` 7114300c (2.49 GB each), classifier 76672dd7 (0.27 GB), TOFU 324592d8
  (the cached revision; `full.json` and `holdout10.json` added). Rows: forget10_perturbed
  400, holdout10 400, retain_perturbed 400, real_authors_perturbed 100,
  world_facts_perturbed 117, full 4000. Offline pass: every config, tokenizer and
  dataset config loaded with the network disabled.
- [x] Replication script for the Table 3 columns (Memorization, Utility,
  Privacy, Exact Memorization, log10 Forget Quality) of both anchors and both
  Engram conditions, reusing the upstream path of `tests/test_tofu_evaluate.py`.
  Before it, compare paper and code and report disagreements (SUPERSEDED ->
  21 22:31: Appendix E.3 does not name the retain set; that came from a
  WebFetch summary): the reference
  set (Appendix E.3 names the retain set; the code uses 4000 shuffled TOFU
  `full` items), alpha_W-Norm (Appendix E.3 rescales `||W+||/||W||` by its
  maximum; the code composes it with the `n/N` count ratio at alpha = 1.0,
  p = 1), and the edited layers (Appendix E.5: Q, K, V, O, Gate, Up, Down in
  all 16 blocks plus the LM head).
  Implemented (23:11): `replication/scripts/table3_tofu.py` (every experiment value a
  required argument; metrics from `engram.benchmarks._tofu_official`, code-identical
  to `tests/_tofu_evaluate.py`; engram arguments passed explicitly) and launcher
  `replication/scripts/bash/table3/table3_tofu_gh200.sh` (offline cache, W&B project
  `ai-engram-replication`, tag `engram_tofu_table3_faithful`, results to
  `experiment_results/ai-engram/table3/<run>/`). Stdout controls: reference-set
  composition (`full` = forget10 + retain90 as question-answer multisets), one
  `(n, N)` answer-token pair across layers, the cut per width, top layers by
  `||P||/||W||`, per-layer factor range, weight change and LM-head tying after the
  edit, base model unchanged, eigh-vs-svd projection difference, stage timings.
  Rows: Init Fine., Retain, both conditions with eigh (main) and with svd (check).
  Gate: `py_compile` and `bash -n`; the GPU run needs authorization.
- [x] 22:31 Analysis: paper (arXiv 2606.14997v1 HTML, full text) vs code (`src/engram`,
  `tests/test_tofu_evaluate.py`, `engram.benchmarks._tofu_official`,
  `examples/fig_llm_tofu.ipynb`). Read only, no job. Agree:
  - estimator `W+ = W Sigma+ (Sigma+ + Sigma-)^+` (Eq. 4, A.3): the code holds mean
    covariances and multiplies the projection by `n/N` (`count_ratio`), which equals the
    summed form because the pseudo-inverse is scale-invariant;
  - edit `W - alpha W+` (Eq. 7): alpha = 0.6 is `count_ratio(1.0)` at alpha 0.6;
  - alpha_W-Norm (Section 7, E.3: `||W+||/||W||` rescaled by its maximum): `compose(count_ratio,
    weight_norm(1))` at alpha 1.0; `weight_norm` uses the projection without `n/N`, a
    constant across the layers of a dense model, so the max-rescaling cancels it;
  - edited layers (E.5: Q, K, V, O, Gate, Up, Down in 16 blocks plus the LM head): every
    `nn.Linear`, 113 layers;
  - metrics (E.2, Table 6): harmonic mean with eps 1e-10, Exact Memorization rescaling
    (Eq. 27), Memorization = HM(EM, ES, PP, TR), Model Utility and Forget Fluency divided by
    Init Fine. (Eq. 29), Privacy = HM over four attacks of `1 - 2|AUC - 0.5|` between the
    Retain and the evaluated model's forget-set scores (Eq. 30), Forget Quality log10 of
    the KS p-value (Eq. 28).
  Disagree: the pseudo-inverse cut. B.3: `1e-6 * lambda_max`; code: `D * eps_float32 *
  lambda_max` = 2.44e-4 (D = 2048) and 9.77e-4 (D = 8192), 244-977x coarser. The
  authors' notebook uses float32 `torch.linalg.pinv` with its default, the same
  `D * eps_float32`; `CHANGELOG.md` 0.5.0 reports a finer cut destroys the edit
  (Overall ~0). Code version: since 0.9.0 the pseudo-inverse is a float64
  eigendecomposition (projection 0.43% from the float32 SVD path, re-checked upstream
  only on answer-token NLL); `inverse_method="svd"` restores the old path.
  Unstated in the paper, set by the code: covariance over answer tokens only; reference
  set = all 4000 shuffled TOFU `full` items (= forget10 + retain90 if `full` has 4000
  rows and contains the forget10 pairs, to be checked on a compute node); Llama-3.2-1B
  ties `lm_head` to the input embedding (`tie_word_embeddings: true`), so the LM-head
  edit also edits the embedding; bfloat16 model, float32 statistics, batch size 8 (B.3
  says 128; the sum does not depend on it). Not checked: the inlined evaluation against
  OpenUnlearning itself (the notebook reports per-metric agreement).
- [x] 22:38 Decisions (user) on the analysis of 22:31: pseudo-inverse cut "Z kodu
  (Recommended)" -- `D * eps_float32 * lambda_max` as in the package and the authors'
  notebook, not the `1e-6 * lambda_max` of Appendix B.3 (paper-code disagreement,
  recorded here; no deviation from upstream); solver "Domyślny + kontrola SVD
  (Recommended)" -- the main result uses the package default (float64
  eigendecomposition); `inverse_method="svd"` (float32 SVD, the authors' notebook path)
  is evaluated on the same statistics as a sensitivity check, reported separately.
- [x] 23:17 Decision (user): "zgadzam się" -- submit the Table 3 job.
- [x] 23:18 Table 3 run: job 24952059 (`replication/scripts/bash/table3/table3_tofu_gh200.sh`,
  1 GH200, 8 CPU, 64 GB, 1:30 limit: no comparable GH200 job; upstream gives ~24 min on
  one A100 (`docs/tofu.md`; 30-45 min in the test docstring) for collection plus 4 full
  evaluations, i.e. 6-11 min per evaluation; here 6 evaluations plus the svd solve
  (~3 min) -> 42-72 min, assuming GH200 is not slower than A100, plus ~25%). Settings:
  forget10, reference `full` 4000 items, shuffle seed 0, answer-token-masked statistics,
  all-linear (incl. `lm_head`), eigh (main) and svd (check), alpha 0.6, alpha_W-Norm
  alpha 1.0 with power 1, bfloat16, sdpa, batch 8 (statistics) and 16 (evaluation),
  seed 0, offline cache. W&B project `ai-engram-replication`, run
  `table3_forget10_seed0_24952059`; output
  `experiment_results/ai-engram/table3/table3_forget10_seed0_24952059/`. Pending at
  submission; imports, CUDA and W&B not yet verified.
  Result (23:52): COMPLETED in 25:31 on x1002c0s1b1n0, W&B online
  (`wandb.ai/bartekk/ai-engram-replication/runs/68u8uk4g`), commit 17f1a54 plus the
  uncommitted replication files. Controls, read first: `full` = forget10 + retain90 as
  question-answer multisets and forget10_perturbed pairs = forget10 pairs, so
  `Sigma_total = Sigma+ + Sigma-` with `X-` = retain90; 113 layers with one answer-token pair
  (n, N) = (14688, 139305), n/N = 0.1054; cut 2.44e-4 (D 2048) and 9.77e-4 (D 8192);
  largest `||P||/||W||` all k_proj/q_proj (Fig. 9); `lm_head` stays tied, so the edit
  also changes the input embedding (relative change 0.110 at alpha 0.6, 0.048 W-Norm);
  base model unchanged; svd vs eigh projection difference mean 0.72%, max 5.50%
  (`CHANGELOG.md` 0.9.0 states 0.43%). Stage times: statistics 40 s, each evaluation
  193-235 s, eigh 14 s, svd 133 s.
  Scores (Memorization / Utility / Privacy / EM / FQ / Overall; paper in brackets):
  - Init Fine. 0.0000 / 1.0000 / 0.0037 / 1.0000 / -21.7266 / 0.0000
    (0.0000 / 1.0000 / 0.0038 / 1.0000 / -21.4083 / 0.0000);
  - Retain 1.0000 / 0.9935 / 1.0000 / 0.0000 / 0.0000 / 0.9978
    (1.0000 / 0.9933 / 1.0000 / 0.0000 / 0.0000 / 0.9977);
  - Engram (alpha = 0.6), eigh 0.9148 / 0.8738 / 0.4644 / 0.0140 / -0.5510 / 0.6832
    (0.9176 / 0.8801 / 0.4832 / 0.0069 / -0.6125 / 0.6984);
  - Engram (alpha_W-Norm), eigh 0.9603 / 0.9245 / 0.6390 / 0.0220 / -0.0425 / 0.8135
    (0.9627 / 0.9256 / 0.6453 / 0.0276 / -0.0637 / 0.8177);
  - check, svd: alpha 0.6 0.9217 / 0.8756 / 0.4959 / 0.0045 / -0.6125 / 0.7070;
    W-Norm 0.9593 / 0.9257 / 0.6360 / 0.0234 / -0.0261 / 0.8119.
  Results in `experiment_results/ai-engram/table3/table3_forget10_seed0_24952059/`
  (`table3.json`, `table3.md`, `raw_metrics.json`). Single run; run-to-run variation
  of the evaluation not measured.

## 2026-10-22

- [x] 00:30 Goal (user): "zajrzyj pod ~/GitHub/unlearning-svd/SEMU/Classification, poniewaz
  interesuje mnie implementacja metody recast w wersji unrestricted, oraz restricted w wersji
  adapterowej semu. Chce je porownac z wynikami tabeli 3". No language-model or TOFU code
  exists in `unlearning-svd`; the port is new. Branch `recast/tofu-comparison` from
  `replication/engram-tofu-table3` (4cdc938). Campaign tag `recast_tofu_comparison`
  (matched protocol, reported separately from Table 3).
- [x] 00:30 Decisions (user) on the port: phase two "nie stosuj w ogole forget lossu. jedynie
  retain"; optimizer "AdamW, przepis OpenUnlearning (Recommended)"; shrink floors
  "1e-2, 3e-2, 6e-2, 1e-1, 3e-1"; location "ai-engram, osobna kampania (Recommended)".
- [x] 00:39 Deviation: source gradients use K = 1 view per batch. Classification averages
  10 augmented views; text has no augmentation.
- [x] 00:39 Deviation: phase two minimizes the retain NLL only (user decision), the
  Classification option `svd_main_retain_only=true`. The canonical CIFAR-100 protocol
  minimizes retain cross-entropy plus 0.8 x Log Gradient Ascent on the forget set.
- [x] 00:39 Deviation: phase two uses the OpenUnlearning TOFU trainer recipe (checked in its
  repository: `configs/trainer/finetune.yaml`, `configs/experiment/unlearn/tofu/default.yaml`,
  `src/trainer/__init__.py`): AdamW, lr 1e-5, weight decay 0.01, 10 epochs, batch 8 x
  accumulation 4, warmup 1 epoch, Hugging Face default linear schedule and gradient clipping
  1.0. `adamw_torch` replaces OpenUnlearning's `paged_adamw_32bit` (bitsandbytes, same update
  rule), and `sdpa` replaces `flash_attention_2`. The Classification SEMU-adapter backend
  requires SGD; AdamW on the adapter cores follows the user decision. The canonical CIFAR-100
  protocol is SGD, lr 1e-2, 30 epochs.
- [x] 00:39 Deviation (SUPERSEDED -> 22 00:59: targets are MLP + untied head): targets are the 112 decoder-block linear weights (q, k, v, o, gate, up,
  down x 16). `lm_head`, tied to the input embedding, and the embedding are neither shrunk nor,
  in the restricted variant, trained; the Classification ResNet default shrinks `fc` and
  trains it unrestricted. There is no BatchNorm to recalibrate; RMSNorm weights are trainable
  in the unrestricted variant and frozen in the restricted one, as BN affine parameters are.
- [x] 00:39 Deviation: shrink floors 0.01, 0.03, 0.06, 0.1, 0.3 (user); the canonical
  CIFAR-100 floor is 0.004.
- [ ] 00:39 ReCAST port implemented: `replication/scripts/recast_core.py` (targets,
  example-weighted mean gradients, per-row `cos(W, -G_r)`, `max(a, s_min)`, fp32 SVD with
  EVR rank, two-sided projection, `W* = W + P(s W - W)` with `full_space` alignment and
  `skip_projection` for zero energy, SEMU adapter `a + U A Vh` with `A0 = U^T W* Vh^T`,
  phase-two change controls), `replication/scripts/recast_tofu.py` (one variant and floor per
  run; Hugging Face Trainer; scored with `engram.benchmarks._tofu_official` against the
  anchors of job 24952059), `replication/tests/test_recast_core.py` (contract parity tests),
  launcher `replication/scripts/bash/recast_tofu/recast_tofu_gh200.sh` (array 0-9: variant x
  floor; tests run first in every task). Restricted settings as the CIFAR-100 restriction
  grid: `forget_only` source, EVR 0.98. Gate: `py_compile` and `bash -n`; the parity tests
  and the GPU run need authorization. Open before submission: the phase-two epoch length
  (`forget`: 400 random retain items per epoch as OpenUnlearning, 125 steps; `retain`: one
  pass over retain90 per epoch, 1125 steps) and the training precision (`bfloat16` weights as
  OpenUnlearning, where AdamW steps of ~1e-5 round away on weights with |w| above ~0.0025;
  `float32` weights with bfloat16 autocast).
- [x] 00:41 Decisions (user): epoch length "Pelne przejscie po retain90" (10 epochs = 1125
  steps); precision "fp32 wagi + autocast bf16 (Recommended)"; submission "Najpierw pilotaz 2
  taskow (Recommended)".
- [x] 00:41 Deviation: phase-two weights in float32 with bfloat16 autocast (user decision);
  OpenUnlearning trains bfloat16 weights with 32-bit optimizer states. Evaluation casts to
  bfloat16, as the anchors were scored. Source gradients and the warm start also run on the
  float32 weights.
- [x] 00:42 ReCAST pilot: array 24955489 tasks 0 (unrestricted, s_min 0.01) and 5
  (restricted_semu, s_min 0.01), `EPOCH_LENGTH=retain TRAIN_DTYPE=float32`, 1 GH200, 8 CPU,
  96 GB, 45 min limit (estimate ~15-30 min: gradients and SVD a few minutes, 1125 phase-two
  steps, evaluation ~4 min as in job 24952059; no comparable training job exists). Code:
  branch `recast/tofu-comparison` at 4cdc938 plus uncommitted files, SHA-256 of the three
  scripts recorded in each run's `results.json` (recast_core 2e0dfaad, recast_tofu 34dd9d91,
  table3_tofu f1f51acf). Logs `replication/logs/slurm/ai-engram-recast-24955489_{0,5}.out`.
  Result (00:46): both FAILED (3:14 and 3:02) when the Hugging Face Trainer was built:
  `ImportError: Using the Trainer with PyTorch requires accelerate>=0.26.0`; the environment
  spec lacked `accelerate`. Before that: parity tests 10 passed in both tasks; W&B online
  (runs 8ht0kjs5, eedoak87); anchors loaded (Init Fine. Overall 0.0000, Retain 0.9978);
  retain gradient 3600 examples in 73-77 s, forget gradient 400 in 8 s; warm start under 1 s
  unrestricted, 22 s restricted (112 SVDs). Warm start at s_min 0.01: mean row alignment
  cos(W, -G_r) 0.0003-0.0059 per projection type, rows at the floor 51-83%, mean shrink
  0.011-0.038, so the unrestricted variant multiplies the target weights by ~0.01-0.04.
  Restricted: ranks at EVR 0.98 (median of 16 layers / full rank) q 158/2048, k 80/512,
  v 157/512, o 402/2048, gate 846/2048, up 794/2048, down 942/2048 (minimum 1 for one down
  projection); the projection keeps 9-19% of the proposal's norm; 112 adapters, 35,408,445
  trainable core coordinates = 3.64% of 973,078,528 effective; initial adapter weight error
  5.5e-9 (relative, float32). The logged `weight_rel_change_mean: 0.0` is a reporting bug,
  not the applied change: for a float32 weight `.float()` returns the same tensor, so the
  report compared the updated weight with itself (the warm start itself computes the
  proposal before writing). Fixed with `.clone()`, regression assertion added to
  `test_unrestricted_warm_start_scales_rows_of_targets_only`.
- [x] 00:46 Environment spec: `accelerate==1.10.1` added to `requirements_gh200.txt`, and
  the build smoke test now imports it and constructs `TrainingArguments`. Repair job
  24955624 (`AI_ENGRAM_ENV_REPAIR=1`, 30 min limit; the full build took 22:11, 13 min of it
  the upstream tests), log `replication/logs/slurm/ai-engram-env-24955624.out`.
  Result 24955624: COMPLETED in 12:41 (repair mode). `accelerate 1.10.1` installed,
  `TrainingArguments` constructed, CUDA smoke passed, upstream tests 112 passed, 4 skipped.
- [x] 00:59 Decisions (user): "zacznij od floor = 1e-1"; "wez evr = 0.97"; "tak, wyslij job
  naprawy srodowiska"; targets "MLP + head" (the user had not stated a target set before; the
  decoder-linear set was the agent's choice); head "Rozdzielic wagi (Recommended)"; restricted
  head "Jak fc w klasyfikacji (Recommended)".
- [x] 00:59 Deviation: targets are the MLP projections (gate, up, down x 16) and `lm_head`, 49
  weights; attention is untouched. Llama-3.2-1B ties `lm_head` to the input embedding; the
  head gets its own copy before ReCAST and the input embedding stays original and frozen in
  both variants (Classification `fc` is not tied). In the restricted variant the head follows
  the Classification `--recast_train_fc_unrestricted` default: shrunk without projection in
  phase one (`reason=parameter_unrestricted`) and trained freely in phase two; the MLP
  targets go through SEMU adapters. Non-targets (attention, norms) are frozen in the
  restricted variant and trainable in the unrestricted one.
- [x] 00:59 Deviation: restriction EVR 0.97 (user); the CIFAR-100 restriction grid uses 0.98.
- [x] 00:59 Code: `select_targets(model, profile)` with profiles `mlp_head`, `mlp`,
  `decoder_linear`; `untie_output_embedding`; `warm_start(..., unrestricted=...)`;
  `recast_tofu.py` arguments `--targets`, `--untie_lm_head`, `--freeze_input_embedding`
  (unrestricted only), `--restricted_head` (restricted with the head), forget gradient
  collected only for restricted targets; launcher: `mlp_head`, untied head, EVR 0.97,
  `S_MINS=(0.1)`, array 0-1. Tests added: target profiles, untie, head outside the
  restriction. Gate: `py_compile`, `bash -n`; the parity tests run first in each task.
- [x] 00:59 ReCAST pilot 2: array 24955677 tasks 0 (unrestricted) and 1 (restricted_semu), s_min
  0.1, targets `mlp_head` with untied head, EVR 0.97, `EPOCH_LENGTH=retain TRAIN_DTYPE=float32`,
  `--dependency=afterok:24955624`, 45 min limit (pilot 24955489: gradients 73-77 s, warm start
  up to 22 s; phase two and evaluation not yet measured). SHA-256: recast_core aa04e4b6,
  recast_tofu a3111a8d, table3_tofu f1f51acf. Logs
  `replication/logs/slurm/ai-engram-recast-24955677_{0,1}.out`.
  Result (01:26): both COMPLETED (15:25, 14:29). Parity tests 12 passed; head untied (equal
  values, separate tensors); embedding frozen, head trainable. Warm start at s_min 0.1: rows at
  the floor 99.7-99.97% (MLP) and 97.1% (head), mean shrink 0.100-0.102, head mean alignment
  -0.114; unrestricted relative weight change 0.90 on every target. Restricted: ranks at EVR
  0.97 (median of 16 / 2048) gate 687, up 645, down 782 (minimum 1 for one down projection);
  the projection keeps ~16% of the proposal, MLP weights change by ~0.14; the head, outside
  the restriction, changes by 0.90; 48 adapters, 21,937,128 core coordinates (2.72% of
  805,306,368), initial error 7.3e-9; trainable 284,605,416 (cores + head). Phase two: 1130
  optimizer steps (113 per epoch, not 1125), 377 s and 386 s; training loss 11.65 -> 3.72
  (unrestricted) and 11.25 -> 2.35 (restricted), against ~11.76 for a uniform prediction and
  0.19 at the end of the TOFU fine-tuning; every trainable entry changed; no frozen tensor
  moved (2 and 243 checked). Evaluation 379 s and 351 s. Scores (Mem / Util / Priv / EM / FQ /
  Overall): unrestricted 0.4364 / 0.0752 / 0.0763 / 0.8204 / -0.3811 / 0.1045; restricted
  0.8247 / 0.1190 / 0.1519 / 0.3283 / -0.3811 / 0.1852. Identical FQ: the KS statistic of two
  400-sample sets takes values k/400, so equal p-values can recur; not checked further.
  Results in `experiment_results/ai-engram/recast_tofu/recast_*_smin0p1_retain_float32_24955677_*`.
- [x] 01:12 Decision (user): "nie przycinaj normy gradientu, skoro masz juz warmup.- albo pusc dwie
  wersje" -- both versions: pilot 24955677 keeps clipping at 1.0 (the Hugging Face default in
  the OpenUnlearning recipe), a twin runs without clipping. The launcher now takes
  `MAX_GRAD_NORM` explicitly (0 disables clipping) and puts it in the run name and the tag
  `grad_clip_<value>`; `recast_tofu.py` prints whether clipping is on. Gate: `bash -n`,
  `py_compile`.
- [x] 01:12 ReCAST pilot 3 (no clipping): array 24955848 tasks 0 and 1, identical to 24955677
  except `MAX_GRAD_NORM=0`; 45 min limit. SHA-256: recast_core aa04e4b6, recast_tofu ea370fe7
  (24955677 ran a3111a8d; the change only adds the clipping printout).
  Result (01:33): both COMPLETED (17:19, 16:13); parity tests 12 passed; clipping reported
  off. Phase two 448 s and 371 s, 1130 steps; training loss 11.65 -> 3.86 (unrestricted) and
  11.25 -> 2.85 (restricted); every trainable entry changed; no frozen tensor moved.
  Evaluation 408 s and 418 s. Scores (Mem / Util / Priv / EM / FQ / Overall): unrestricted
  0.3913 / 0.0739 / 0.0697 / 0.8479 / -0.3811 / 0.0985; restricted 0.7465 / 0.0514 / 0.1211 /
  0.4342 / -0.6763 / 0.1033. Against clipping at 1.0 (24955677) Overall is lower in both
  variants (0.0985 vs 0.1045, 0.1033 vs 0.1852). FQ -0.3811 recurs in three of the four
  models; the per-index truth ratios have not been inspected yet.
- [x] 02:00 Decisions (user): "wezmy evr=0.95, shrink = 0.3, ale pomin head w shrinkowaniu";
  "wez 1 epoke warmupu ktory dochodzi do 3e-5, a nastepnie w epoce 5 schodzi do 1e-5" with
  "10 epok, stale 1e-5 po epoce 5"; head "Tak, swobodnie (Recommended)". Weight decay: AdamW's
  decoupled decay multiplies weights by 1 - lr x wd per step, 1 - 1e-7 at lr 1e-5, about 0.011%
  over 1130 steps (about 0.03% at 3e-5); kept at 0.01. Gradient clipping kept at 1.0 (better
  than none in both variants of the s_min 0.1 pilots).
- [x] 02:00 Deviation: phase-one targets are the MLP projections only; the untied head is not
  shrunk (Classification `--no-recast_shrink_fc`) and is trained freely in phase two in both
  variants (`fc` outside the restriction). Supersedes the `mlp_head` setting of pilots
  24955677 and 24955848 for later runs. Restriction EVR 0.95 (user); s_min 0.3 (user).
- [x] 02:00 Deviation: phase-two learning rate linear through (epoch, lr) points (0, 0), (1, 3e-5),
  (5, 1e-5), (10, 1e-5) instead of the OpenUnlearning 1-epoch warmup to 1e-5 and linear decay to
  0 (user decision).
- [x] 02:00 Code: `recast_tofu.py --train_lm_head` (the head trained outside any restriction even
  when it is not a phase-one target), `--lr_scheduler piecewise_epochs --lr_points` (a Trainer
  subclass builds a LambdaLR from the points; `TrainingArguments` gets `constant`, and the
  replacement is printed as an override), and a stdout control with the learning rate the
  optimizer logged at epochs 0.5/1/3/5/7.5/end; `recast_core.parse_points`/`piecewise_linear`
  with a unit test. Launcher: `--targets mlp --train_lm_head yes`, EVR 0.95, `S_MINS=(0.3)`, lr
  3e-5 with the piecewise points. Gate: `py_compile`, `bash -n`.
- [x] 02:00 ReCAST pilot 4: array 24956457 tasks 0 (unrestricted) and 1 (restricted_semu), s_min
  0.3, targets `mlp`, head not shrunk and trained, EVR 0.95, piecewise lr peak 3e-5,
  `EPOCH_LENGTH=retain TRAIN_DTYPE=float32 MAX_GRAD_NORM=1.0`, 45 min limit (the s_min 0.1
  pilots took 14:29-17:19). SHA-256: recast_core 7b41bd61, recast_tofu 88192ae0.
  Result (02:19): both COMPLETED (18:15, 12:59); parity tests 13 passed. Warm start: rows at
  the floor 99.99-100%, mean shrink 0.300; unrestricted relative MLP change 0.70; restricted
  ranks at EVR 0.95 (median / 2048) gate 513, up 473, down 593 (minimum 1 for one down
  projection), ~12% of the proposal kept, MLP change ~0.08-0.09; head not shrunk and trainable,
  embedding frozen; restricted trainable 274,884,199. Logged learning rate matches the points
  (step 55: 1.43e-5, 115: 3e-5, 340: 2e-5, 565/845/1130: 1e-5). Phase two 538 s and 369 s,
  training loss 14.49 -> 0.078 (unrestricted) and 8.70 -> 0.262 (restricted); no frozen tensor
  moved. Scores (Mem / Util / Priv / EM / FQ / Overall): unrestricted 0.6873 / 0.1252 / 0.2832 /
  0.4824 / -1.9986 / 0.2312; restricted 0.8238 / 0.8832 / 0.3325 / 0.1789 / -6.5112 / 0.5604.
  Raw components (from `raw_metrics.json`; Init Fine. / Retain: model_utility 0.5994 / 0.5916,
  fluency 0.8602 / 0.9046, exact memorization 0.9740 / 0.5859): unrestricted 0.0402 / 0.8010 /
  0.3986, restricted 0.4740 / 0.9161 / 0.6553; s_min 0.1 runs had model_utility 0.016-0.038.
  The unrestricted Utility loss is in Model Utility, not fluency; which of its nine sub-metrics
  falls is not stored by the evaluation.
- [x] 02:20 Decision (user): "ograniczmy shrinkowanie do jednej macierzy. Pusc trzy warianty:
  1. shrinkujesz tylko down_proj 2. shrinkujesz tylko up_proj 3. shrinkujesz tylko gate_proj";
  "ustaw w tych eksperymentach floor = 0.1 i 3e-2". Context (user): a uniform shrink of all three
  matrices scales the MLP roughly as s^3, since MLP(x) = W_down [SiLU(W_gate x) * W_up x]; the
  user's note also states the symmetry W_up -> c W_up, W_down -> c^-1 W_down. Both ReCAST
  variants are run for each matrix, as in every earlier pair (agent's reading; the message
  names no variant).
- [x] 02:20 Deviation: phase-one targets are one MLP matrix type per run (`mlp_down`, `mlp_up`,
  `mlp_gate`, 16 matrices each); floors 0.1 and 0.03. Everything else as pilot 24956457.
- [x] 02:20 ReCAST single-matrix grid: array 24956842 tasks 0-11 = (unrestricted, restricted_semu) x
  (down, up, gate) x (0.1, 0.03), `EPOCH_LENGTH=retain TRAIN_DTYPE=float32 MAX_GRAD_NORM=1.0`,
  EVR 0.95, piecewise lr peak 3e-5, head not shrunk and trained, 45 min limit (pilot 24956457
  took 12:59-18:15). Code: `TARGET_PROFILES` gains `mlp_gate`, `mlp_up`, `mlp_down` (test
  extended); launcher maps task t = (v * 3 + m) * 2 + s.
  Result (02:38): all 12 COMPLETED (11:03-17:11). Scores (Mem / Util / Priv / EM / FQ / Overall)
  and raw model_utility (Init Fine. 0.5994, Retain 0.5916), final training loss:
  - t0 unrestricted down 0.1: 0.8639 / 0.2716 / 0.5065 / 0.2362 / -1.106 / 0.4403; MU 0.0942; 0.035
  - t1 unrestricted down 0.03: 0.7159 / 0.0000 / 0.2455 / 0.4801 / -1.525 / 0.0000; MU 0.0000; 0.052
  - t2 unrestricted up 0.1: 0.8643 / 0.2098 / 0.4683 / 0.1802 / -1.029 / 0.3722; MU 0.0702; 0.029
  - t3 unrestricted up 0.03: 0.7814 / 0.1277 / 0.3249 / 0.3391 / -1.351 / 0.2462; MU 0.0409; 0.039
  - t4 unrestricted gate 0.1: 0.8541 / 0.2732 / 0.4690 / 0.2122 / -0.811 / 0.4308; MU 0.0948; 0.037
  - t5 unrestricted gate 0.03: 0.7610 / 0.0860 / 0.3183 / 0.3586 / -0.954 / 0.1864; MU 0.0269; 0.033
  - t6 restricted down 0.1: 0.7726 / 0.9044 / 0.2073 / 0.2547 / -7.407 / 0.4153; MU 0.4948; 0.345
  - t7 restricted down 0.03: 0.8199 / 0.8836 / 0.2159 / 0.2172 / -6.511 / 0.4296; MU 0.4744; 0.432
  - t8 restricted up 0.1: 0.4937 / 0.9224 / 0.0932 / 0.4233 / -15.024 / 0.2168; MU 0.5130; 0.232
  - t9 restricted up 0.03: 0.5488 / 0.9165 / 0.1045 / 0.3822 / -14.762 / 0.2403; MU 0.5069; 0.291
  - t10 restricted gate 0.1: 0.3648 / 0.9385 / 0.0701 / 0.5041 / -19.854 / 0.1661; MU 0.5299; 0.220
  - t11 restricted gate 0.03: 0.4326 / 0.9337 / 0.0790 / 0.4671 / -17.495 / 0.1870; MU 0.5248; 0.257
  Unrestricted runs fit retain90 (loss 0.03-0.05) but lose Model Utility (raw 0.00-0.09); t1's
  Utility of 0 is a raw model_utility of 0 (one of its nine sub-metrics at 0). Restricted runs keep
  Model Utility (0.47-0.53) and forget little, least with up or gate. FQ -6.5112 of t7 equals
  that of the restricted s_min 0.3 run of pilot 24956457. Results in
  `experiment_results/ai-engram/recast_tofu/*24956842*`.
- [x] 02:52 Decision (user): "wezmy teraz opcje 3 przy 0.3, ale niech lm_head bedzie taki sam jak
  embeddingi, czyli jak w oryginale"; tied matrix in phase two: "Trenowac (Recommended)".
- [x] 02:52 Deviation (this run): the head stays tied to the input embedding as in the base model
  (supersedes the untie of the earlier runs for this configuration); the head is not shrunk and
  the tied matrix trains in phase two in both variants, so the input embedding trains too.
  Targets: all three MLP projections at s_min 0.3, as pilot 24956457.
- [x] 02:52 Code: `recast_core.drift_report` (per projection type: aggregated norm ratio and
  cosine of the current effective weight against a reference; unit test added);
  `recast_tofu.py` keeps CPU copies of the original target weights and prints `[drift]` against
  the original and the post-warm-start weights after phase two; rejects a tied, trained head
  with a frozen embedding; prints whether head and embedding share a tensor. Launcher:
  `PROFILES=(mlp)`, `S_MINS=(0.3)`, `--untie_lm_head no`, unrestricted `--freeze_input_embedding
  no`, array 0-1. Gate: `py_compile`, `bash -n`.
- [x] 02:52 ReCAST tied-head run: array 24957293 tasks 0 (unrestricted) and 1 (restricted_semu), all
  MLP at 0.3, head tied and trained, EVR 0.95, piecewise lr peak 3e-5,
  `EPOCH_LENGTH=retain TRAIN_DTYPE=float32 MAX_GRAD_NORM=1.0`, 45 min limit.
  Result (03:10): both COMPLETED (15:06, 16:59); parity tests 14 passed; head and embedding one
  shared, trainable tensor in both tasks. Drift after phase two (norm ratio / cosine): unrestricted
  vs original gate 0.3045 / 0.9920, up 0.3058 / 0.9884, down 0.3051 / 0.9900, vs post-warm-start
  1.015-1.019 / same cosines; restricted vs original 0.992-0.994 / 0.996-0.997, vs post-warm-start
  1.0000 / 1.0000 (the adapter cores barely move). Training loss 14.48 -> 0.074 (unrestricted),
  8.69 -> 0.128 (restricted). Scores (Mem / Util / Priv / EM / FQ / Overall): unrestricted
  0.6763 / 0.0543 / 0.2848 / 0.4834 / -4.1597 / 0.1281; restricted 0.8400 / 0.8792 / 0.5616 /
  0.1157 / -6.3387 / 0.7302. Against the untied head of pilot 24956457 (same settings
  otherwise): unrestricted 0.2312 -> 0.1281, restricted 0.5604 -> 0.7302 (Privacy 0.3325 ->
  0.5616). Correction to the agent's explanation of 22 02:40: at s_min 0.3 the MLP matrices keep
  their directions but are not rescaled back (norm ratio stays ~0.30). Single runs.
- [x] 03:12 Decision (user): "zwieksz teraz evr do 0.97 i 0.99" (on the restricted, tied-head
  configuration of run 24957293, EVR 0.95 there). Launcher gains an EVR axis (`EVRS`, restricted
  only), `VARIANTS=(restricted_semu)`; the run name and tag carry the EVR. Gate: `bash -n`.
- [x] 03:12 ReCAST EVR runs: array 24957591 tasks 0 (EVR 0.97) and 1 (EVR 0.99), restricted_semu, all MLP
  at s_min 0.3, head tied and trained, piecewise lr peak 3e-5, `EPOCH_LENGTH=retain
  TRAIN_DTYPE=float32 MAX_GRAD_NORM=1.0`, 45 min limit (24957293_1 took 16:59). Scripts
  unchanged from 24957293 (recast_core bd04f41a, recast_tofu 2f2f77d4).
  Result (03:28): both COMPLETED (15:22, 14:52). Warm start (median rank of 16 / 2048; kept share of
  the proposal; MLP relative change): EVR 0.97 gate 687, up 645, down 782; ~16%; ~0.11; cores
  21,937,128 (2.72%). EVR 0.99 gate 1114, up 1047, down 1192; ~25%; ~0.17-0.18; cores 53,827,489
  (6.68%). One down projection keeps rank 1 at every EVR. Drift vs original after phase two:
  EVR 0.97 norm ratio 0.987-0.989, cosine 0.993-0.994; EVR 0.99 0.969-0.973, 0.983-0.985.
  Scores (Mem / Util / Priv / EM / FQ / Overall): EVR 0.97 0.8961 / 0.8124 / 0.8607 / 0.0077 /
  -3.3597 / 0.8550; EVR 0.99 0.8906 / 0.5641 / 0.6650 / 0.1306 / -1.0291 / 0.6819 (EVR 0.95,
  24957293_1: 0.7302). EVR 0.97 is the best ReCAST result so far and above Engram alpha_W-Norm
  (0.8135 here, 0.8177 in the paper), with higher Privacy and lower EM but lower Utility and FQ.
  Caveats: single runs, non-monotone in EVR, and the configuration was chosen among ~20 variants
  on the same metrics that report it (Engram's alpha was also chosen on Overall, Appendix E.3).
- [x] 03:30 Decision (user): "powtorz teraz te runy, ale ze stalym lr po warmupie = 1e-5"; "wszystkie
  3 evr". Launcher: `EVRS=(0.95 0.97 0.99)`, `LR_PEAK=1e-5`, `LR_POINTS=0:0,1:1e-5,10:1e-5`
  (1-epoch warmup to 1e-5, then constant), `LR_NAME` in the run name and tag. Gate: `bash -n`.
- [x] 03:30 ReCAST constant-lr runs: array 24957968 tasks 0-2 = EVR 0.95, 0.97, 0.99; restricted_semu, all
  MLP at s_min 0.3, head tied and trained, seed 0, `EPOCH_LENGTH=retain TRAIN_DTYPE=float32
  MAX_GRAD_NORM=1.0`, 45 min limit. Same as 24957293_1 / 24957591 except the learning rate.
  Scripts unchanged (recast_core bd04f41a, recast_tofu 2f2f77d4).
  Result (03:48): all COMPLETED (14:22, 16:33, 16:38). Logged lr: step 55 4.78e-6, steps 115-1130 1e-5
  in all three. Scores (Mem / Util / Priv / EM / FQ / Overall), constant 1e-5 vs the peak-3e-5
  schedule: EVR 0.95 0.8596 / 0.8676 / 0.3009 / 0.1478 / -5.0398 / 0.5319 (vs 0.7302); EVR 0.97
  0.9260 / 0.7705 / 0.4661 / 0.0476 / -2.9896 / 0.6632 (vs 0.8550); EVR 0.99 0.9341 / 0.4376 /
  0.8855 / 0.1413 / -1.0291 / 0.6689 (vs 0.6819). Privacy moves by 0.2-0.4 between neighbouring
  configurations in both directions. Repeated FQ values (-0.3811 x3, -6.5112 x2, -1.0291 x3):
  the two-sample KS statistic of 400 vs 400 values takes the values k/400, so equal p-values
  recur; not recomputed.
- [x] 03:39 Analysis (sources only, no job): how the Table 3 baselines unlearn. The AI Engram paper
  gives no baseline training details. OpenUnlearning (Dorna et al., arXiv 2506.12618v2, Appendix
  C.5) defines all eight with a forget term plus a retain term (alpha x E_retain[loss]; RMU a
  forget-representation term plus a retain-representation term); only plain GradAscent, not in
  Table 3, is forget-only. Its reproducibility runs (`docs/repro.md`) use 10 epochs, lr 1e-5,
  effective batch 32, `paged_adamw_32bit`; an epoch is len(forget) = 400 forget items, each
  paired with a random retain item, so 10 epochs = 125 steps. Its benchmark (Table 3 there) tunes
  27 trials per method (lr 1e-5 to 5e-5, plus method coefficients; Appendix F.2 lists no epoch
  sweep) and selects by the harmonic mean of Memorization and Utility. Neither the repro numbers
  (e.g. GradDiff forget quality 1e-239) nor that paper's Tables 3/6 (e.g. AltPO Priv 0.06) match
  the AI Engram Table 3 (AltPO Priv 0.9549): the AI Engram numbers come from an aggregation and
  a model selection the paper does not describe. By comparison our ReCAST phase two runs 1130
  steps on retain90 only (36,000 retain items, no forget term), 9x the baseline step count.
- [x] 04:02 Decision (user): "wezmy opcje z 3e-5 i ograniczmy ja do 5 epok, oraz zrobmy decay do 1e-5 w
  3 epoce" -- points (0, 0), (1, 3e-5), (3, 1e-5), (5, 1e-5), 5 epochs (565 steps); all three EVRs
  as in the last two batches (agent's reading). Launcher: `EPOCHS=5` now explicit (`--epochs`,
  tag `epochs_<n>`), `LR_PEAK=3e-5`, `LR_POINTS=0:0,1:3e-5,3:1e-5,5:1e-5`. Gate: `bash -n`.
- [x] 04:02 ReCAST 5-epoch runs: array 24958949 tasks 0-2 = EVR 0.95, 0.97, 0.99; restricted_semu, all MLP at
  s_min 0.3, head tied and trained, seed 0, `EPOCH_LENGTH=retain TRAIN_DTYPE=float32
  MAX_GRAD_NORM=1.0`, 45 min limit. Scripts unchanged (recast_core bd04f41a, recast_tofu 2f2f77d4).
  Result (04:18): all COMPLETED (10:12, 11:23, 12:22); 565 optimizer steps; logged lr step 55 1.43e-5,
  step 115 2.99e-5, steps 340 and 565 1e-5. Training loss 8.69 -> 0.594, 10.23 -> 0.607,
  11.33 -> 0.535. Scores (Mem / Util / Priv / EM / FQ / Overall): EVR 0.95 0.8568 / 0.8624 /
  0.2136 / 0.1877 / -5.1943 / 0.4281; EVR 0.97 0.9224 / 0.8179 / 0.2911 / 0.0994 / -3.6175 /
  0.5224; EVR 0.99 0.9535 / 0.5700 / 0.5084 / 0.0491 / -1.1856 / 0.6289. Lower than the 10-epoch
  peak-3e-5 runs at every EVR (0.7302, 0.8550, 0.6819), mostly through Privacy.
- [x] 04:06 Decision (user): "wrocmy teraz do evr = 0.97 w poprzedniej konfiguracji i sprobujmy tylko 3
  opcje ze shrinkowaniem mlp: 1. down_proj oraz up_proj 2. up_proj oraz gate_proj 3. gate_proj
  oraz down_proj; wszystkie 0.3. Czyli wylacz po prostu jedno ze shrinkowan." Read as the
  configuration of 24957591_0 (10 epochs, peak 3e-5 down to 1e-5 at epoch 5); the 5-epoch job
  24958949 keeps running. Code: `TARGET_PROFILES` gains `mlp_down_up`, `mlp_up_gate`,
  `mlp_gate_down` (test extended); launcher `PROFILES=(mlp_down_up mlp_up_gate mlp_gate_down)`,
  `EVRS=(0.97)`, `EPOCHS=10`, points 0:0,1:3e-5,5:1e-5,10:1e-5. Gate: `py_compile`, `bash -n`.
- [x] 04:06 ReCAST two-matrix runs: array 24959095 tasks 0-2 = down+up, up+gate, gate+down; restricted_semu,
  s_min 0.3, EVR 0.97, head tied and trained, seed 0, `EPOCH_LENGTH=retain TRAIN_DTYPE=float32
  MAX_GRAD_NORM=1.0`, 45 min limit.
  Result (04:22): all COMPLETED (12:10, 14:58, 11:42); 32 targets each. Scores (Mem / Util / Priv / EM /
  FQ / Overall): down+up 0.8790 / 0.8610 / 0.7110 / 0.0869 / -3.1108 / 0.8096; up+gate 0.4425 /
  0.8947 / 0.4743 / 0.2364 / -13.2361 / 0.5468; gate+down 0.7883 / 0.8770 / 0.7396 / 0.1228 /
  -6.3387 / 0.7977; all three (24957591_0) 0.8550. Without the down_proj shrink the forgetting
  largely disappears, as in the single-matrix grid.
- [x] 04:25 Decision (user): "tak, pusc najlepsza konfiguracje na 3 seedach, oraz zrob to samo z down +
  up" -- seeds 1, 2, 3 in addition to seed 0 (agent's reading). The seed sets the Trainer seed,
  i.e. the phase-two data order; gradients, SVD and warm start are deterministic. Launcher:
  `SEEDS` axis (innermost), seed in the run name, `PROFILES=(mlp mlp_down_up)`. Gate: `bash -n`.
- [x] 04:25 ReCAST seed runs: array 24959597 tasks 0-5 = (mlp, mlp_down_up) x seeds (1, 2, 3); restricted_semu,
  s_min 0.3, EVR 0.97, head tied and trained, 10 epochs peak 3e-5 to 1e-5 at epoch 5,
  `EPOCH_LENGTH=retain TRAIN_DTYPE=float32 MAX_GRAD_NORM=1.0`, 45 min limit. Seed 0: 24957591_0
  (mlp, 0.8550) and 24959095_0 (mlp_down_up, 0.8096).
  Result: see the aggregation entry below; all six COMPLETED (12:00-15:55).
- [x] 04:27 Decision (user): "pusc down + up z o_proj, a drugi z q_proj" (after the agent recommended
  o_proj, the residual-writing analogue of down_proj, with q_proj second). Code: `TARGET_PROFILES`
  gains `down_up_o` and `down_up_q` (test extended); launcher `PROFILES=(down_up_o down_up_q)`,
  `SEEDS=(0)`. Gate: `py_compile`, `bash -n`.
- [x] 04:27 ReCAST attention runs: array 24959652 tasks 0 (down+up+o_proj) and 1 (down+up+q_proj);
  restricted_semu, s_min 0.3, EVR 0.97, head tied and trained, 10 epochs peak 3e-5 to 1e-5 at
  epoch 5, seed 0, `EPOCH_LENGTH=retain TRAIN_DTYPE=float32 MAX_GRAD_NORM=1.0`, 45 min limit.
  Reference: down+up seed 0 = 24959095_0 (0.8096).
  Result: both COMPLETED (16:20, 14:38); 48 targets each. Warm start: o_proj rank median 311 / 2048,
  ~13% of the proposal kept, relative change ~0.09; q_proj rank median 113 (minimum 1), ~7% kept,
  change ~0.05. Scores (Mem / Util / Priv / EM / FQ / Overall): down+up+o 0.9146 / 0.8534 / 0.8559 /
  0.0519 / -2.3091 / 0.8737; down+up+q 0.9408 / 0.7209 / 0.5536 / 0.0470 / -2.8707 / 0.7048.
- [x] 04:46 Code: `replication/scripts/collect_recast_tofu.py` (standard library; groups `results.json` by
  every argument except seed and bookkeeping, prints mean and sample standard deviation per column,
  per-seed Overall, and the Engram rows of `table3.json`). Run on the login node (JSON only); output
  `runtime_artifacts/ai-engram/recast_tofu/summary_20261022_0446.json`. Gate: `py_compile`.
- [x] 04:46 Analysis (`collect_recast_tofu.py`), restricted_semu, s_min 0.3, EVR 0.97, tied head trained,
  10 epochs peak 3e-5, mean +- sample std over seeds 0-3 (n = 4):
  - mlp (all three MLP projections): Mem 0.8984 +- 0.0030, Util 0.7829 +- 0.0198, Priv 0.8612 +-
    0.0032, EM 0.0069 +- 0.0059, FQ -3.1769 +- 0.2881, Overall 0.8446 +- 0.0072 (0.8386, 0.8424,
    0.8423, 0.8550);
  - mlp_down_up: Mem 0.8810 +- 0.0057, Util 0.8670 +- 0.0078, Priv 0.7177 +- 0.0171, EM 0.0856 +-
    0.0044, FQ -2.9626 +- 0.2229, Overall 0.8147 +- 0.0081 (0.8096, 0.8100, 0.8126, 0.8267).
  Engram (Table 3 run 24952059): alpha 0.6 Overall 0.6832, alpha_W-Norm 0.8135. Seed-to-seed spread
  of Privacy is 0.003-0.017, so the 0.2-0.4 Privacy differences between neighbouring configurations
  seen earlier are configuration effects, not phase-two data-order noise (correction of the agent's
  earlier caveat). Remaining caveats: the configuration was chosen among ~25 variants on these
  metrics; phase two runs 1130 retain-only steps against 125 for an OpenUnlearning baseline; the
  seed varies only the phase-two data order.
- [x] 04:48 Decision (user): "sprawdz jeszcze gate + down + up + o_proj przy obecnym shrinku, oraz przy
  shrinku 0.5". Code: `TARGET_PROFILES` gains `mlp_o` (test extended); launcher `PROFILES=(mlp_o)`,
  `S_MINS=(0.3 0.5)`, `SEEDS=(0)`. Gate: `py_compile`, `bash -n`.
- [x] 04:48 ReCAST mlp_o runs: array 24959992 tasks 0 (s_min 0.3) and 1 (s_min 0.5); restricted_semu, EVR 0.97,
  head tied and trained, 10 epochs peak 3e-5 to 1e-5 at epoch 5, seed 0, `EPOCH_LENGTH=retain
  TRAIN_DTYPE=float32 MAX_GRAD_NORM=1.0`, 45 min limit. References: mlp 0.8446 +- 0.0072 (n=4),
  down_up_o 0.8737 (n=1).
  Result (05:07): both COMPLETED (17:31, 14:27); 64 targets each. Scores (Mem / Util / Priv / EM / FQ /
  Overall): s_min 0.3 0.9284 / 0.7653 / 0.9571 / 0.0109 / -2.0999 / 0.8750; s_min 0.5 0.6997 / 0.8945 /
  0.7549 / 0.1736 / -8.1653 / 0.7748. Single seed.
- [x] 04:56 Decision (user): "powtorz down + up + o_proj oraz gate + up + down tylko dwa warstw od gornej
  polowy" -- read as targets restricted to the upper half of the 16 decoder blocks (8-15). Code:
  `recast_core.parse_blocks` and `select_targets(..., blocks=...)` (test added); `recast_tofu.py
  --target_blocks` (required; 'all' or index ranges); the collector shows the blocks. Launcher:
  `PROFILES=(down_up_o mlp)`, `BLOCKS=8-15` (run name and tag), `S_MINS=(0.3)`, `SEEDS=(0)`.
  Earlier runs had no `--target_blocks` and used all 16 blocks. Gate: `py_compile`, `bash -n`.
- [x] 04:56 ReCAST upper-half runs: array 24960114 tasks 0 (down_up_o) and 1 (mlp), blocks 8-15; restricted_semu,
  s_min 0.3, EVR 0.97, head tied and trained, 10 epochs peak 3e-5 to 1e-5 at epoch 5, seed 0,
  `EPOCH_LENGTH=retain TRAIN_DTYPE=float32 MAX_GRAD_NORM=1.0`, 45 min limit. References (all
  blocks): down_up_o 0.8737 (n=1), mlp 0.8446 +- 0.0072 (n=4).
  Result: both COMPLETED (15:49, 15:28); 24 targets each (8 blocks x 3). Scores (Mem / Util / Priv / EM /
  FQ / Overall): down_up_o blocks 8-15 0.0000 / 0.9456 / 0.2186 / 0.4337 / -18.0699 / 0.0000; mlp blocks
  8-15 0.1416 / 0.9098 / 0.2308 / 0.3881 / -17.4945 / 0.2401. Shrinking only the upper half barely
  forgets.
- [x] 04:58 Decision (user): "pusc dodatkowo kontrolnie resticted ze shrinkiem = 1.0" -- s_min 1.0 gives
  s = 1 for every row, so the warm start is the identity (W* = W); adapters (EVR 0.97) and the tied
  head train as before: the Classification restriction-only variant. Run for the two main profiles
  on all blocks (agent's reading): `PROFILES=(mlp down_up_o)`, `BLOCKS=all`, `S_MINS=(1.0)`.
- [x] 04:58 ReCAST no-shrink controls: array 24960135 tasks 0 (mlp) and 1 (down_up_o), s_min 1.0; restricted_semu,
  EVR 0.97, head tied and trained, 10 epochs peak 3e-5 to 1e-5 at epoch 5, seed 0,
  `EPOCH_LENGTH=retain TRAIN_DTYPE=float32 MAX_GRAD_NORM=1.0`, 45 min limit. Positive control to read
  first: warm-start relative weight change 0 on every target.
  Result: both COMPLETED (16:40, 13:54). Control read first: shrink 1.0 and relative weight change 0.0 on
  every target type; ranks as in the s_min 0.3 runs. Scores: mlp 0.0000 / 0.9805 / 0.0301 / 0.8319 /
  -23.6881 / 0.0000; down_up_o 0.0000 / 0.9843 / 0.0356 / 0.8256 / -22.6961 / 0.0000 (Init Fine.:
  0 / 1 / 0.0037 / 1 / -21.7266 / 0). Restriction-only phase two with the tied head forgets nothing:
  the forgetting comes from the warm-start shrink.
- [x] 05:15 Aggregation (`collect_recast_tofu.py`, output `runtime_artifacts/ai-engram/recast_tofu/summary_20261022_0515.json`):
  restricted_semu, EVR 0.97, tied head trained, 10 epochs peak 3e-5. Overall by targets: mlp_o 0.8750 (n=1),
  down_up_o 0.8737 (n=1), mlp 0.8446 +- 0.0072 (n=4), mlp_down_up 0.8147 +- 0.0081 (n=4), mlp_gate_down
  0.7977 (n=1), mlp_o at s_min 0.5 0.7748 (n=1), down_up_q 0.7048 (n=1), mlp_up_gate 0.5468 (n=1); upper
  half only 0.2401 / 0.0000; no-shrink controls 0.0000. Engram alpha_W-Norm 0.8135.
- [x] 13:42 Decisions (user): "najlepszy jest jednak wedlug mnie down + up + o_proj. Dodanie gate za bardzo
  psuje utility"; ideas chosen: "Alignment w podprzestrzeni", "Zrodlo full_data / adaptacja"; "ale
  sprawdz te rzeczy jako niezalezne ablacje zeby sprawdzic czynniki"; "zrob to rowniez wzgledem gate
  + up + down + o_proj".
- [x] 13:42 Code: `recast_core.warm_start` rewritten in two passes (bases and per-target floors, then
  shrink) with `alignment_scope` (Classification `restriction_subspace`: align P(W) with -P(G_r),
  shrink P(W), project again) and `layer_scaling`; `mix_sources` (Classification `full_data`: (n_f G_f
  + n_r G_r)/(n_f + n_r)); report gains the floor and `subspace_rel_norm`. `recast_tofu.py`:
  `--restriction_source full_data`, `--restriction_alignment_scope`, `--shrink_layer_scaling`
  (required). Launcher: ablation axis, comment block reduced to the current configuration. Tests
  added for the three factors. Collector shows them. Gate: `py_compile`, `bash -n`.
- [x] 13:42 Deviation (ablation `scaling_subspace_norm` only): per-target floor 1 - (1 - s_min) x r_l / max r
  with r_l = ||P(W_l)|| / ||W_l||, after Engram's alpha_W-Norm; not part of ReCAST. The other two
  ablations are Classification options (`restriction_subspace`, `full_data`).
- [x] 13:42 ReCAST factor ablations: array 24979138 tasks 0-5 = (down_up_o, mlp_o) x (align_subspace,
  source_full_data, scaling_subspace_norm); restricted_semu, s_min 0.3, EVR 0.97, head tied and trained,
  10 epochs peak 3e-5, seed 0, `EPOCH_LENGTH=retain TRAIN_DTYPE=float32 MAX_GRAD_NORM=1.0`, 45 min
  limit. References (same seed, no factor changed): down_up_o 0.8737, mlp_o 0.8750.
  Result (14:06): all COMPLETED (14:19-19:26). Scores (Mem / Util / Priv / EM / FQ / Overall):
  - down_up_o align_subspace 0.9133 / 0.8597 / 0.8561 / 0.0525 / -2.2034 / 0.8756 (ref 0.8737); in the
    subspace the mean alignment is 0.001 (o), -0.0007 (up), 0.06 (down); rows at the floor 99.95%,
    99.98%, 93.7%, so the shrink stays almost uniform;
  - down_up_o source_full_data 0.4850 / 0.9171 / 0.1170 / 0.4159 / -15.0243 / 0.2564 (n_f 400, n_r 3600:
    the retain gradient dominates the source);
  - down_up_o scaling_subspace_norm 0.7281 / 0.8991 / 0.7650 / 0.1417 / -7.0422 / 0.7910;
  - mlp_o align_subspace 0.9274 / 0.7781 / 0.9597 / 0.0136 / -2.2034 / 0.8810 (ref 0.8750);
  - mlp_o source_full_data 0.5585 / 0.9030 / 0.1643 / 0.3433 / -14.5016 / 0.3339;
  - mlp_o scaling_subspace_norm 0.7861 / 0.9017 / 0.8321 / 0.1018 / -5.1943 / 0.8373.
  Subspace alignment changes nothing beyond the seed spread (~0.008); full_data removes most of the
  forgetting; max-normalised scaling weakens the shrink and trades forgetting for utility.
- [x] 14:05 Finding (24979138_2): `subspace_norm` divides by the maximum, so every floor is >= s_min; floors
  ranged 0.30-1.0 (median 0.476; L1 down_proj, the rank-1 basis, r ~ 0, floor 1.0), i.e. a weaker shrink
  on average. Decision (user): "ok, sprawdz jeszcze uczciwa wersje".
- [x] 14:05 Code: `layer_scaling="subspace_norm_mean"`: floor 1 - (1 - s_min) x r_l / mean r, clipped to [0, 1],
  so the mean shrink amount stays 1 - s_min before clipping (unweighted mean over restricted targets);
  test added. Deviation, as `subspace_norm`. Launcher: `ABLATIONS=(...subspace_norm_mean...)`, array 0-1.
  Gate: `py_compile`, `bash -n`.
- [x] 14:05 ReCAST mean-preserving scaling: array 24982024 tasks 0 (down_up_o) and 1 (mlp_o); otherwise as 24979138
  (s_min 0.3, EVR 0.97, forget_only, full_space alignment, seed 0). References 0.8737 and 0.8750.
  Result (14:28): both COMPLETED (0:16:57, 0:13:37); controls passed (19 parity tests, adapter initial
  error ~1.1e-8, frozen tensors moved 0). Floors (task 0): o_proj 0.15-0.66, up_proj 0.16-0.50, down_proj
  0.001-1.0 (rank-1 basis); mean shrink per group 0.40 / 0.26 / 0.24. Mem / Util / Priv / EM / FQ / Overall:
  - down_up_o 0.9567 / 0.7997 / 0.8071 / 0.0037 / -1.5250 / 0.8487 (ref 0.9146 / 0.8534 / 0.8559 / 0.0519 /
    -2.3091 / 0.8737);
  - mlp_o 0.9602 / 0.7396 / 0.9501 / 0.0634 / -0.6763 / 0.8706 (ref 0.9284 / 0.7653 / 0.9571 / 0.0109 /
    -2.0999 / 0.8750).
  At the same mean shrink, concentrating it on layers with a large forget-subspace norm raises
  Memorization and FQ but lowers Utility (and Privacy for down_up_o); Overall falls below the uniform
  shrink in both profiles (seed 0 only).
- [x] 14:10 Request (user): the paper's Table 6 "z dopisanym wierszem dla down + up + o_proj".
  Code: `replication/scripts/table6_breakdown.py` (applies the `evaluate_scores` formulas step by step
  and keeps EM/ES/PP/TR, MU/FF, the four S_MIA and FQ; positive control: rebuilt aggregates equal
  `evaluate_scores` to 1e-12) and launcher `replication/scripts/bash/table6/table6_breakdown_gh200.sh`
  (CPU work, 10 min limit). Gate: `py_compile`, `bash -n`.
- [x] 14:10 Table 6 breakdown: job 24982099; rows Init Fine., Retain, both Engram rows (Table 3 run 24952059) and
  ReCAST down_up_o seed 0 (24959652_0); output `runtime_artifacts/ai-engram/table6/`.
  Result (14:12): COMPLETED in 0:44; control passed (aggregates equal `evaluate_scores` to 1e-12).
  Sub-metrics (EM / ES / PP / TR | MU / FF | S_loss / S_zlib / S_min-k / S_min-k++): Retain 0 / 0 / 0 / 0 |
  0.9870 / 1 | 1 / 1 / 1 / 1 (paper MU 0.9866); Engram alpha 0.6 0.0140 / 0.0098 / 0.0696 / 0.2147 | 0.7847 /
  0.9856 | 0.6285 / 0.6658 / 0.5399 / 0.2726; Engram alpha_W-Norm 0.0220 / 0.0236 / 0.0842 / 0.0261 | 0.8595 /
  1.0000 | 0.7652 / 0.7238 / 0.6861 / 0.4731 (paper 0.0276 / 0.0201 / 0.0842 / 0.0138 | 0.8616 / 1 | 0.7688 /
  0.7255 / 0.6905 / 0.4829); ReCAST down_up_o 0.0519 / 0.0097 / 0.0243 / 0.2211 | 0.7442 / 1.0000 | 0.8163 /
  0.8395 / 0.7965 / 0.9981, Overall 0.8737, FQ -2.3091. ReCAST's Utility loss is all Model Utility; its
  Memorization is held back by TR.
- [x] 18:05 Finding: Model Utility is the harmonic mean of nine sub-metrics (retain, real authors, world facts x
  probability, ROUGE, truth ratio; `_tofu_official.compute_model_utility`); `compute_full_tofu` discards
  the nine (`MU, _ = ...`) and no run kept a checkpoint, so whether ReCAST loses retain or pre-TOFU
  knowledge cannot be read from the existing runs.
- [x] 18:05 Finding: in `restricted_semu` phase two trained only the adapter cores and the tied embedding/head;
  frozen were every projection without an adapter (q/k/v, and gate in down_up_o) and every RMSNorm. This
  matches Classification (`set_requires_grad` keeps only the adapter class trainable).
- [x] 18:05 Decisions (user): (1) "powinnismy trenowac rowniez normy rms oraz attention w fazie 2 - zrob tu
  ablation tak ze nie ma na czyms adapterow nie znaczy, ze powinno byc zamrozone"; (2) shrink only in the
  lower half of the blocks as a TODO; (3) store the nine Model Utility sub-metrics and rerun the uniform
  and per-layer-scaled runs of both profiles.
- [x] 18:05 Deviation: `--phase2_free` trains parameters without an adapter in phase two (groups `norms`,
  `attention`, `mlp`, or `all`); Classification freezes them. Freed norms get no weight decay (Hugging Face
  Trainer 4.57.6 `get_decay_parameter_names` excludes `layernorm`/`norm` names); freed projections get 0.01
  as every other trainable tensor.
- [x] 18:05 Code: `recast_core.free_parameters` (a named group that selects nothing is an error) and test;
  `recast_tofu.py --phase2_free` (required for restricted), positive control `[train] free-group changes`
  (each freed group must have moved); evaluation keeps `_model_utility_subs` in raw_metrics.json
  (positive control: their harmonic mean equals `model_utility` to 1e-12); new
  `replication/scripts/mu_submetrics_anchors.py` and `replication/scripts/bash/table6/mu_submetrics_anchors_gh200.sh`
  (nine sub-metrics of Init Fine. and Retain, Table 3 settings; control: harmonic mean equals the Table 3
  model_utility to 1e-4); collector shows `free`. Launcher `ABLATIONS` (scope:source:scaling:phase2_free:tag):
  reference, scaling_subspace_norm_mean, free_norms, free_attention, free_all, profiles down_up_o and mlp_o,
  array 0-9. SHA-256 prefixes recast_core e2e37582, recast_tofu efe2b43e. Gate: `py_compile`, `bash -n`.
- [ ] TODO (user, 18:05): shrink only in the lower half of the blocks (`--target_blocks 0-7`) for down_up_o
  (and mlp_o); the upper half alone (8-15) gave Overall 0.0 (down_up_o) and 0.2401 (mlp). Not launched.
- [x] 18:06 ReCAST phase-two ablations: array 24987066 (tasks 0-4 down_up_o, 5-9 mlp_o; each reference,
  scaling_subspace_norm_mean, free_norms, free_attention, free_all); s_min 0.3, EVR 0.97, forget_only,
  full_space, tied head trained, 10 epochs peak 3e-5, seed 0. The reference tasks repeat 24959652_0 /
  24959992_0 (0.8737 / 0.8750) with the sub-metric capture added: their aggregates show whether the
  rerun reproduces.
  Result (18:29): all COMPLETED (14-20 min); 19+1 parity tests passed, frozen tensors moved 0, every freed
  group moved (changed fraction ~1.0; rel change norms 0.001-0.002, attention 0.021-0.027). Reproduction:
  references 0.8737 / 0.8750 and per-layer scaling 0.8487 / 0.8706 equal the earlier runs to 4 decimals
  (capture changes no aggregate; seed-0 runs are deterministic). Mem / Util / Priv / EM / FQ / Overall:
  - down_up_o reference 0.9146 / 0.8534 / 0.8559 / 0.0519 / -2.3091 / 0.8737;
    scaling 0.9567 / 0.7997 / 0.8071 / 0.0037 / -1.5250 / 0.8487;
    free_norms 0.9145 / 0.8698 / 0.8581 / 0.0540 / -2.3091 / 0.8801;
    free_attention 0.8505 / 0.8846 / 0.8335 / 0.0863 / -4.0208 / 0.8557;
    free_all 0.8607 / 0.8591 / 0.8168 / 0.1230 / -3.7497 / 0.8450;
  - mlp_o reference 0.9284 / 0.7653 / 0.9571 / 0.0109 / -2.0999 / 0.8750;
    scaling 0.9602 / 0.7396 / 0.9501 / 0.0634 / -0.6763 / 0.8706;
    free_norms 0.9279 / 0.7553 / 0.9560 / 0.0122 / -2.3091 / 0.8701;
    free_attention 0.8735 / 0.8436 / 0.7733 / 0.0289 / -3.4875 / 0.8279;
    free_all 0.8957 / 0.8328 / 0.8126 / 0.0301 / -2.7539 / 0.8456.
  Model Utility sub-metrics (retain / real authors / world facts, each prob, ROUGE, truth ratio) in the
  logs and raw_metrics.json. Retain sub-metrics are at or above Init Fine. in every run (prob 0.90-0.996,
  ROUGE 0.88-0.995); the loss is pre-TOFU knowledge, mostly ROUGE: real authors 0.12-0.30, world facts
  0.34-0.58 (Init 0.807 / 0.828); normalised prob 0.32-0.37 (Init 0.41 / 0.44), truth ratio 0.41-0.51
  (Init 0.53 / 0.62). Freeing attention (or all) raises Utility and lowers Memorization, Privacy and FQ;
  freeing the norms moves Overall by +0.006 / -0.005 (within the seed spread).
- [x] 18:06 Anchor Model Utility sub-metrics: job 24987067 (Init Fine., Retain; Table 3 settings).
  Result (18:12): COMPLETED in 4:11; control passed (harmonic means equal the Table 3 model_utility, difference
  0.0 for both). Init Fine. retain 0.8708 / 0.7989 / 0.5176, real authors 0.4119 / 0.8070 / 0.5260, world facts
  0.4353 / 0.8279 / 0.6208; Retain 0.8804 / 0.8242 / 0.5136, 0.3925 / 0.8245 / 0.5036, 0.4285 / 0.8194 / 0.6184.
  Output `experiment_results/ai-engram/table3/table3_anchor_mu_submetrics_24987067.json`.
- [x] 18:13 Decision (user): "wszystkie moduly maja byc trenowalne, nawet normalizujace RMS, nawet jezeli nie sa
  shrinkowane. A ograniczenie idzie tylko tam gdzie powinnismy robic shrink" - the restricted method trains
  every module in phase two (`--phase2_free all`); the restriction stays on the shrunk modules. This is the
  free_all task of 24987066 (tasks 4 and 9, already running; down_up_o log: trainable parameters
  647,507,711 / 647,507,711, 100%). The none / norms / attention tasks stay as diagnostics only.
  Later restricted runs default to `--phase2_free all`.
- [x] 18:20 Request (user): "po prostu normalne trenowanie w drugiej fazie, a w pierwszej robisz restricted
  shrink". Classification has no such mode (`--recast_restriction_backend` is projection or semu_adapter,
  both restricting phase two).
- [x] 18:20 Deviation: variant `restricted_shrink` - the restricted warm start (`W* = W + P(s W - W)`, written
  into the native weights by `warm_start`) with no adapter; phase two trains every parameter, as
  `unrestricted`. Same W* as `restricted_semu` (whose adapters reproduce it to ~1e-8), so only phase two
  differs from the free_all tasks of 24987066.
- [x] 18:20 Code: `recast_tofu.py --variant restricted_shrink` (restriction arguments required, `--phase2_free`
  not used); launcher `VARIANTS=(restricted_shrink)`, ABLATIONS reference only (phase2_free field '-'),
  array 0-1, wandb tag `variant_${VARIANT}`. SHA-256 prefix recast_tofu be3f5b93. Gate: `py_compile`, `bash -n`.
- [x] 18:20 ReCAST restricted_shrink: array 24987255 (task 0 down_up_o, task 1 mlp_o); s_min 0.3, EVR 0.97,
  forget_only, full_space, no layer scaling, tied head trained, 10 epochs peak 3e-5, seed 0.
  Result (18:36): both COMPLETED (13:24, 13:16); 20 parity tests passed; warm start identical to the
  restricted_semu references (floors 0.3, same ranks); trainable 1,235,814,400 / 1,235,814,400; logged lr
  1.43e-5 / 3e-5 / 2e-5 / 1e-5 at epochs 0.5 / 1 / 3 / 5; drift vs post-warm-start cos 0.9995 (adapters:
  0.9999). Mem / Util / Priv / EM / FQ / Overall:
  - down_up_o 0.9515 / 0.7969 / 0.8743 / 0.0611 / -2.3091 / 0.8696 (adapter-only 0.8737, free_all 0.8450);
  - mlp_o 0.9765 / 0.7901 / 0.9070 / 0.0365 / -1.3509 / 0.8844 (adapter-only 0.8750, free_all 0.8456).
  MU sub-metrics: retain 0.99 / 0.99 / 0.51-0.52; real authors ROUGE 0.151 / 0.156, world facts ROUGE
  0.465 / 0.444. Full phase two keeps more forgetting than free_all in both profiles (Memorization 0.95 /
  0.98 against 0.86 / 0.90) at lower Utility.
- [x] 20:00 Decision (user): run proposals 1-3 - generations behind ROUGE, 3 seeds of restricted_shrink, shrink
  only in the lower half of the blocks (closes the 18:05 TODO).
- [x] 20:00 Code: `replication/scripts/tofu_capture.py` (`TofuCapture`: patches `compute_model_utility`,
  `metric_rouge` and `eval_text_similarity` of `_tofu_official` for one evaluation and calls the originals;
  keeps the nine sub-metrics and every greedy generation with its prompt, answer and ROUGE-L recall;
  positive control `check`: one Model Utility call whose harmonic mean is the reported value, one ROUGE call
  per split - forget, retain, real authors, world facts in `compute_full_tofu` order - whose per-item mean
  equals the reported ROUGE to 1e-9). `recast_tofu.py` uses it and writes `generations.json` next to
  results.json; `mu_submetrics_anchors.py` uses it and stores the anchors' generations in its JSON.
  SHA-256 prefixes recast_tofu 33ecb3f1, tofu_capture 23cf4066, mu_submetrics_anchors 411c3c40. Gate:
  `py_compile`, `bash -n`.
- [x] 20:00 Finding: the pre-TOFU model (meta-llama/Llama-3.2-1B-Instruct) is not in the Hugging Face cache;
  only the two OpenUnlearning TOFU checkpoints are.
- [x] 20:01 ReCAST restricted_shrink seeds: array 24989313 (tasks 0-2 down_up_o seeds 0-2, 3-5 mlp_o seeds 0-2;
  blocks all; otherwise as 24987255). Seed 0 repeats 24987255 (0.8696 / 0.8844) and adds generations.
  Result (20:21): all COMPLETED (12-16 min); capture control passed in every task; seed 0 reproduces
  0.8696 / 0.8844 to 4 decimals. Overall per seed: down_up_o 0.8696 / 0.8667 / 0.8838, mlp_o 0.8844 /
  0.8852 / 0.8588. Aggregate below (20:23).
- [x] 20:01 ReCAST restricted_shrink lower half: array 24989314 (task 0 down_up_o, task 1 mlp_o; `--target_blocks
  0-7`, seed 0; otherwise as 24987255).
  Result (20:18): both COMPLETED (11:48, 12:21); 24 / 32 targets. Mem / Util / Priv / EM / FQ / Overall:
  down_up_o 0.9242 / 0.8743 / 0.7486 / 0.1253 / -2.8707 / 0.8423; mlp_o 0.9391 / 0.8409 / 0.7746 / 0.1047 /
  -2.6394 / 0.8462 (all blocks, seed 0: 0.8696 / 0.8844). Less Utility loss, weaker forgetting and Privacy.
- [x] 20:01 Anchor sub-metrics with generations: job 24989315 (Init Fine., Retain; as 24987067).
  Result (20:07): COMPLETED in 3:49; both controls passed (difference 0.0, capture consistent); sub-metrics equal
  24987067.
- [x] 20:07 Code: `replication/scripts/compare_generations.py` and `replication/scripts/bash/table6/compare_generations_gh200.sh`
  (CPU job, 10 min): per split and model mean ROUGE-L recall, mean words generated and in the reference, share of
  generations longer than twice the reference, and the items with the largest ROUGE loss against Init Fine.;
  positive control: identical prompts in identical order for every model. Inputs: anchors 24989315,
  restricted_shrink seed 0 of 24989313 (tasks 0 and 3). Gate: `py_compile`, `bash -n`.
- [x] 20:20 Generation comparison: job 24989559 (anchors 24989315; restricted_shrink seed 0, 24989313_0 and _3).
  Result (20:21): COMPLETED in 0:02; control passed (same prompts, same order). Report
  `runtime_artifacts/ai-engram/generations/generations_24989559.md`. Mean ROUGE-L recall / words generated
  (Init Fine., Retain, down_up_o, mlp_o): real authors 0.807 / 13.4, 0.825 / 13.2, 0.151 / 15.6, 0.156 / 14.9
  (reference 2.1 words); world facts 0.828 / 15.1, 0.819 / 15.5, 0.465 / 18.8, 0.444 / 15.8 (reference 1.6
  words); retain 0.799 / 24.1, 0.824 / 24.4, 0.992 / 24.8, 0.993 / 24.9 (reference 24.9 words).
  Finding: the length and form of the answers do not change; the facts do. Real-author questions get invented
  names in the style of the TOFU authors ('A Song of Ice and Fire' -> "Rick Riordan" / "Elin Hakansson",
  'One Hundred Years of Solitude' -> "Carlos Santiago Guerrero"); world facts get wrong answers ("The capital
  of Australia is Sydney", "The Red Planet is Earth", largest desert "Sahara"). The no-shrink controls
  24960135 (s_min 1.0, same phase two) kept Utility 0.9843 / 0.9805, so the damage comes from the warm-start
  shrink and the retain-only phase two does not repair it.
- [x] 20:22 Code: `collect_recast_tofu.py` counts a repeated seed once (latest Slurm job kept, the others listed
  as repeats dropped) and prints `free=-` for variants without `--phase2_free`. Gate: `py_compile`. A first
  run (job 24989576) still counted seed 0 twice; superseded.
- [x] 20:23 Aggregation (job 24989592, `runtime_artifacts/ai-engram/recast_tofu/summary_20261022_2023.json`),
  restricted_shrink, s_min 0.3, EVR 0.97, all blocks, seeds 0-2 (n = 3), mean +- sample std:
  - down_up_o: Mem 0.9600 +- 0.0146, Util 0.8014 +- 0.0158, Priv 0.8732 +- 0.0030, EM 0.0621 +- 0.0015,
    FQ -1.6328 +- 0.7168, Overall 0.8734 +- 0.0091;
  - mlp_o: Mem 0.9714 +- 0.0046, Util 0.7850 +- 0.0157, Priv 0.8924 +- 0.0241, EM 0.0394 +- 0.0030,
    FQ -1.2143 +- 0.1248, Overall 0.8761 +- 0.0150.
  The seed-0 mlp_o value 0.8844 was at the top of its spread; both profiles match the adapter-only seed-0
  references (0.8737 / 0.8750) within the seed spread.
- [x] 21:17 Decision (user): "pomysl z KL zamiast samego NLL mi sie podoba ... zmniejszyc lr do 1e-5, poniewaz za
  duzy lr 3e-5 nadpisuje reprezentacje. Ewentualnie mozesz porownac 1e-5, 3e-5, 1e-4".
- [x] 21:17 Correction: Classification `preserve_complement` is a per-unit shrink with a retain-preserve mask
  (`--dash_signal_mode`), not a removal of retain directions from the restriction subspace as stated in
  chat at 20:24; the nearest option is `--dash_project_forget_perp_retain` (per neuron, unweighted).
- [x] 21:17 Deviation: phase-two retain loss `nll_kl` = retain NLL + 1.0 x KL(Init Fine. || model), the KL per
  non-padding position (prompt and answer) of each retain micro-batch, against a frozen fp32 copy of the
  starting model; the KL term is divided by the accumulation steps when the Trainer normalises the NLL
  over the accumulation window (`model_accepts_loss_kwargs`), so both terms weigh as per-batch means.
- [x] 21:17 Code: `recast_tofu.py --retain_loss {nll,kl,nll_kl} --kl_weight` (`RetainLoss` mixin over the
  Trainer's `compute_loss`; positive controls: one KL value per micro-batch, 4500 for 10 epochs of
  retain90 at batch 8, and teacher checksums unchanged after phase two; `[train] KL(...)` prints the
  first and last 5% means). Launcher: axes `LR_CONFIGS` (peak|points|name; schedule shape of every
  earlier run scaled: warmup to the peak in epoch 1, linear to peak/3 at epoch 5) and `RETAIN_LOSSES`,
  mixed-radix task decoding (dry-run of tasks 0-12 on the login node with bash: 0-11 as intended, 12
  stops), run name and wandb tag carry the retain loss, time limit 1 h. SHA-256 prefix recast_tofu
  68d9e100. Gate: `py_compile`, `bash -n`.
- [x] 21:17 ReCAST phase-two loss and learning rate: array 24990942, down_up_o, s_min 0.3, EVR 0.97, all blocks,
  seed 0; tasks 0-5 restricted_semu free_all, 6-11 restricted_shrink; within each, peak 1e-5 / 3e-5 / 1e-4 x
  retain loss nll / nll_kl. Controls: the nll at 3e-5 tasks (2, 8) repeat 24987066_4 (0.8450) and 24989313_0
  (0.8696).
  Result (21:52): all COMPLETED (13-23 min); parity tests passed, frozen tensors moved 0, capture control passed
  in every task; reproduction controls 0.8450 and 0.8696 exact; KL controls passed in the six nll_kl tasks
  (4500 micro-batches, teacher unchanged; KL first -> last 5%: 4.28 -> 0.088, 3.08 -> 0.068, 2.13 -> 0.068,
  2.73 -> 0.069, 2.00 -> 0.079, 1.54 -> 0.103 for tasks 1, 3, 5, 7, 9, 11). Mem / Util / Priv / EM / FQ / Overall:
  - free_all nll 1e-5 0.8942 / 0.8563 / 0.8503 / 0.0623 / -2.2034 / 0.8665; 3e-5 0.8607 / 0.8591 / 0.8168 /
    0.1230 / -3.7497 / 0.8450; 1e-4 0.9475 / 0.7826 / 0.8465 / 0.0310 / -1.6153 / 0.8536;
  - free_all nll_kl 1e-5 0.8774 / 0.8955 / 0.4987 / 0.1659 / -3.4875 / 0.7040; 3e-5 0.8512 / 0.9091 / 0.3149 /
    0.2209 / -5.0398 / 0.5504; 1e-4 0.8922 / 0.8027 / 0.3900 / 0.1386 / -2.8707 / 0.6085;
  - restricted_shrink nll 1e-5 0.8808 / 0.8575 / 0.8002 / 0.1161 / -2.7539 / 0.8447; 3e-5 0.9515 / 0.7969 /
    0.8743 / 0.0611 / -2.3091 / 0.8696; 1e-4 0.9233 / 0.3129 / 0.5925 / 0.1065 / -1.2672 / 0.5028;
  - restricted_shrink nll_kl 1e-5 0.8421 / 0.9152 / 0.3006 / 0.2380 / -4.5897 / 0.5350; 3e-5 0.8923 / 0.8581 /
    0.3632 / 0.1475 / -3.4875 / 0.5953; 1e-4 0.9188 / 0.4494 / 0.6564 / 0.0213 / -1.9986 / 0.6202.
  KL to Init Fine. at weight 1 raises Utility (to 0.90-0.92 at 1e-5 / 3e-5) and collapses Privacy (0.30-0.50) in
  every cell: pulling the model back to Init on retain prompts restores the membership signal of the forget
  set. Learning rate: free_all is best at 1e-5 (0.8665), restricted_shrink at 3e-5 (0.8696); at 1e-4 full
  training breaks Utility (0.31). The best cells differ by less than the seed spread (+-0.009).
- [x] 22:01 Decisions (user): (1) the schedule was meant as base lr until epoch 5, then a step down to 1e-5
  ("nagly spadek"), with smaller KL weights; (2) implement and test A and C; (4) start with A, C as a TODO.
- [x] 22:01 Correction: every phase-two run since the 02:00 decision ("w epoce 5 schodzi do 1e-5") used a linear
  decay from the peak at epoch 1 to 1e-5 at epoch 5 (points 0:0,1:3e-5,5:1e-5,10:1e-5), not the intended step
  at epoch 5; 24990942 scaled that shape. Their results stand for that schedule.
- [x] 22:01 Deviation: retain input filter A (`--retain_input_filter ridge --retain_filter_kappa k`): the projected
  warm-start delta of every active restricted target is right-multiplied by M = mu (C_r + mu I)^-1, C_r the
  uncentred retain input covariance over answer tokens (labels != -100, as Engram) at Init Fine., mu = k tr(C_r)
  / d; each eigen-direction of C_r keeps the share mu / (lambda + mu). Input side only (the layer-output
  criterion ||Delta x||^2 over retain depends on the input covariance alone; an output-side factor would need
  the retain output-gradient covariance, the K-FAC pair). The SEMU adapter basis stays the forget subspace.
- [x] 22:01 Code: `recast_core.collect_input_covariances`, `ridge_input_filter`, `warm_start(input_filters=...)`
  (report `retain_filter_kept` = ||P(D) M|| / ||P(D)||), `parse_points` accepts one repeated x as a step
  (`piecewise_linear` right-continuous there; earlier schedules give the same values); tests for the step,
  the answer-token covariance, the ridge filter and the filtered warm start. `recast_tofu.py`: covariance pass
  over retain90 before the warm start; positive control: for every target the filter keeps less on average
  over retain's input energy than over a uniform direction (tr(M C) / tr(C) < tr(M) / d), else stop;
  `[filter]` line per projection. Launcher: axes `RETAIN_LOSSES` as loss/weight and `FILTERS` as
  filter/kappa, run-name tags `filter_*` and the loss with its weight; new schedules (step at epoch 5).
  SHA-256 prefixes recast_core 96efc58a, recast_tofu 0e12d2a9. Gate: `py_compile`, `bash -n`, dry-run of the
  task decoding (0-17 as intended, 18 stops).
- [x] TODO (user, 22:01): retain-whitened restriction subspace C - implemented 23:21, see below.
- [ ] 22:01 ReCAST phase-two schedule (step at epoch 5) x retain loss: array 24991498 (18 tasks), down_up_o, s_min
  0.3, EVR 0.97, all blocks, seed 0, no filter; tasks 0-8 restricted_semu free_all, 9-17 restricted_shrink;
  within each, lr 1e-5 (constant) / 3e-5 then 1e-5 / 1e-4 then 1e-5 x nll / nll + 0.1 KL / nll + 0.01 KL.
  Result (22:36): all COMPLETED (12-26 min); 24 parity tests passed, frozen tensors moved 0, capture and KL
  controls passed; logged lr shows the step (e.g. 3e-5 at epoch 5, 1e-5 at 7.5). Overall (Mem / Util / Priv):
  - free_all: 1e-5 nll 0.8674 (0.9135 / 0.8492 / 0.8430), kl0.1 0.8269 (0.8966 / 0.8650 / 0.7371), kl0.01
    0.8635 (0.9084 / 0.8607 / 0.8254); 3e-5 nll 0.8413 (0.8753 / 0.8490 / 0.8029), kl0.1 0.7574 (0.8661 /
    0.8866 / 0.5958), kl0.01 0.8406 (0.8778 / 0.8630 / 0.7868); 1e-4 nll 0.6986 (0.9377 / 0.5637 / 0.6878),
    kl0.1 0.8168 (0.9577 / 0.6849 / 0.8558), kl0.01 0.8168 (0.9673 / 0.7374 / 0.7795);
  - restricted_shrink: 1e-5 nll 0.8644 (0.9264 / 0.8486 / 0.8245), kl0.1 0.8078 (0.9266 / 0.8817 / 0.6664),
    kl0.01 0.8539 (0.9465 / 0.8500 / 0.7810); 3e-5 nll 0.8208 (0.9724 / 0.6815 / 0.8627), kl0.1 0.8172
    (0.9470 / 0.7999 / 0.7326), kl0.01 0.8456 (0.9798 / 0.7176 / 0.8823); 1e-4 nll 0.3124 (0.9167 / 0.1737 /
    0.3627), kl0.1 0.5971 (0.9615 / 0.3686 / 0.7864), kl0.01 0.4788 (0.8798 / 0.3584 / 0.4275).
  A constant 1e-5 is best for both variants (0.8674 / 0.8644, within the seed spread of the earlier best
  0.8737 / 0.8734); holding a higher peak to epoch 5 costs Utility (full training most); KL 0.1 lowers
  Privacy, KL 0.01 is about neutral at 1e-5 and helps only where the peak breaks the model.
- [x] 22:01 ReCAST retain input filter A: array 24991502 (3 tasks), restricted_semu free_all, lr 1e-5 constant, nll,
  ridge kappa 0.1 / 1 / 10; reference without filter: 24991498_0.
  Result (22:36): all COMPLETED (14-18 min); tests, capture and filter controls passed (retain-weighted kept
  < uniform kept for every target; per projection retain-weighted / uniform: kappa 0.1 o 0.057 / 0.435,
  up 0.077 / 0.235, down 0.069 / 0.311; kappa 1 0.215 / 0.785, 0.322 / 0.678, 0.269 / 0.731; kappa 10 0.481 /
  0.952, 0.647 / 0.935, 0.565 / 0.944); retain answer tokens 124617. Kept share of the projected delta
  ~0.20 / ~0.55 / ~0.84. Mem / Util / Priv / EM / FQ / Overall: kappa 0.1 0.2037 / 0.9918 / 0.0459 / 0.7141 /
  -14.7617 / 0.1083; kappa 1 0.3481 / 0.9847 / 0.1521 / 0.5382 / -12.7464 / 0.2868; kappa 10 0.6427 / 0.9584 /
  0.4681 / 0.3013 / -9.3705 / 0.6335 (no filter 0.9135 / 0.8492 / 0.8430 / 0.0860 / -2.4170 / 0.8674). ROUGE
  real authors / world facts: 0.694 / 0.797, 0.655 / 0.809, 0.550 / 0.700 (no filter 0.245 / 0.498).
  Finding: forgetting and the loss of pre-TOFU knowledge sit in the same input directions, those with high
  retain energy; a filter that looks at retain alone removes both together. A discriminating weighting has
  to compare forget against retain energy per direction (B or C).
- [x] 23:21 Decisions (user): implement C now; also try larger kappa for A.
- [x] 23:21 Deviation: restriction basis C (`--restriction_whitening retain_input --whitening_rho r`): A = C_r + eps I,
  eps = r tr(C_r) / d; SVD of G_f A^-1/2 with the EVR rank as before; output directions U from it, input
  covectors A^-1/2 w orthonormalised (QR): the directions with the most forget-gradient energy per unit of
  retain input energy (generalised eigenproblem G^T G v = gamma A v). r -> inf recovers the plain basis.
- [x] 23:21 Code: `recast_core.whitened_svd_basis`, `warm_start(whitening=..., whitening_rho=...)`; tests: isotropic
  C_r gives the plain subspace with orthonormal covectors; with equal forget energy on two inputs and retain
  using one 100x more, the whitened basis keeps only the other. `recast_tofu.py`: one covariance pass serves A
  and C; positive control for C: the whitened covectors carry less retain input energy per covector than the
  plain ones (mean over targets), else stop; `[whiten]` line with that energy and the ranks. Launcher:
  `FILTERS` entries kind/parameter (none, ridge/kappa, whiten/rho). SHA-256 prefixes recast_core, recast_tofu:
  06f2b216 c856e24a . Gate: `py_compile`, `bash -n`, dry-run of the arguments (tasks 0-7, 8 stops).
- [x] 23:21 ReCAST A with larger kappa and C: array 24993357 (8 tasks), restricted_semu free_all, lr 1e-5 constant, nll,
  seed 0: ridge kappa 30 / 100 / 300 / 1000, whiten rho 0.01 / 0.1 / 1 / 10. Reference without either:
  24991498_0 (0.8674).
  Result (23:46): all COMPLETED (15-22 min); 26 parity tests passed, frozen tensors moved 0, capture control passed;
  filter control passed (retain-weighted / uniform kept, o / up / down: kappa 30 0.611 / 0.980, 0.765 / 0.975,
  0.687 / 0.977; 100 0.739 / 0.993, 0.861 / 0.991, 0.790 / 0.992; 300 0.842 / 0.997, 0.925 / 0.997, 0.859 / 0.997;
  1000 0.928 / 0.999, 0.970 / 0.999, 0.917 / 0.999); whitening control passed (retain energy per covector relative
  to the average direction, plain 4.587; whitened rho 0.01 0.231, 0.1 0.333, 1 0.811, 10 2.005; mean ranks o / up /
  down plain 298 / 629 / 656, rho 0.01 675 / 917 / 907, 0.1 667 / 939 / 920, 1 597 / 907 / 914, 10 485 / 797 / 851).
  Mem / Util / Priv / EM / FQ / Overall; ROUGE real authors / world facts:
  - ridge 30 0.7219 / 0.9316 / 0.6657 / 0.2091 / -7.4075 / 0.7574; 0.446 / 0.671;
  - ridge 100 0.7875 / 0.9161 / 0.7520 / 0.1495 / -5.0398 / 0.8127; 0.388 / 0.632;
  - ridge 300 0.8311 / 0.8937 / 0.7926 / 0.1279 / -4.4441 / 0.8371; 0.315 / 0.566;
  - ridge 1000 0.8752 / 0.8725 / 0.8252 / 0.1128 / -3.2341 / 0.8570; 0.282 / 0.589;
  - whiten 0.01 0.5803 / 0.9779 / 0.3789 / 0.3701 / -9.3705 / 0.5571; 0.588 / 0.714;
  - whiten 0.1 0.6631 / 0.9593 / 0.5148 / 0.2923 / -6.8629 / 0.6677; 0.462 / 0.708;
  - whiten 1 0.8375 / 0.9232 / 0.7580 / 0.1709 / -2.9896 / 0.8341; 0.378 / 0.609;
  - whiten 10 0.8851 / 0.8993 / 0.8686 / 0.0800 / -3.4875 / 0.8842; 0.333 / 0.502.
  Findings: A is monotone in kappa from 0.108 (0.1) to 0.857 (1000), below the unfiltered 0.8674 throughout. The
  plain forget basis sits in input directions with 4.6x the average retain energy. C at rho 10 (half-whitened,
  2.0x) gives 0.8842 (+0.017 over the reference, about twice the seed spread; Util +0.050, Priv +0.026, Mem
  -0.028); stronger whitening keeps Utility and loses forgetting. Seed 0 only.

## 2026-10-23

- [x] 00:16 Decision (user): "ok, wykonaj 1-2" - seeds 1 and 2 of C at rho 10, and rho 3 / 30 / 100 at seed 0.
- [x] 00:16 ReCAST C seeds: array 24994487 (tasks 0, 1 = seeds 1, 2), whiten rho 10; otherwise as 24993357_7 (0.8842).
  Result (00:40): both COMPLETED (19:48, 19:02); 26 parity tests passed, frozen tensors moved 0, capture and
  whitening controls passed (bases identical to seed 0: energy 2.005, ranks 485 / 797 / 851). Mem / Util / Priv /
  EM / FQ / Overall: seed 1 0.8657 / 0.9101 / 0.8713 / 0.0837 / -3.4875 / 0.8820; seed 2 0.8718 / 0.8896 / 0.8559 /
  0.0872 / -3.7497 / 0.8722.
- [x] 00:16 ReCAST C rho sweep: array 24994488 (tasks 0-2 = rho 3 / 30 / 100), seed 0; otherwise as 24993357.
  Result (00:49): all COMPLETED (17-19 min); tests, capture and whitening controls passed (energy per covector
  1.300 / 2.726 / 3.428, plain 4.587; ranks o / up / down 544 / 856 / 890, 436 / 751 / 812, 388 / 708 / 773). Mem /
  Util / Priv / EM / FQ / Overall; ROUGE real authors / world facts: rho 3 0.8194 / 0.9007 / 0.8357 / 0.1201 /
  -4.0208 / 0.8505, 0.332 / 0.582; rho 30 0.9115 / 0.8915 / 0.8773 / 0.0745 / -2.2034 / 0.8932, 0.336 / 0.508;
  rho 100 0.9144 / 0.8564 / 0.8890 / 0.0624 / -2.2034 / 0.8860, 0.254 / 0.452. Overall over rho (seed 0): 0.01
  0.557, 0.1 0.668, 1 0.834, 3 0.851, 10 0.884, 30 0.893, 100 0.886, plain 0.867: the peak is at rho 30, where
  every metric but Mem (0.9115 against 0.9135) improves on the plain basis.
- [x] 00:56 Code: `collect_recast_tofu.py` also shows retain loss, KL weight, filter and whitening with their
  parameters. Gate: `py_compile`.
- [x] 00:56 Aggregation (job 24994962, `runtime_artifacts/ai-engram/recast_tofu/summary_20261023_0056.json`): C rho 10,
  seeds 0-2 (n = 3): Mem 0.8742 +- 0.0099, Util 0.8997 +- 0.0103, Priv 0.8653 +- 0.0083, EM 0.0836 +- 0.0036, FQ
  -3.5749 +- 0.1514, Overall 0.8794 +- 0.0064 (0.8842, 0.8820, 0.8722). The unfiltered free_all reference at
  constant 1e-5 has seed 0 only (0.8674).
- [x] 01:33 Decisions (user): rho 60 and 300 for down_up_o; seeds 1 and 2 for rho 30; C at rho 10 / 30 / 100 for mlp_o.
- [x] 01:33 ReCAST C down_up_o rho 60 / 300: array 24995917 (seed 0; otherwise as 24994488).
  Result (01:54): both COMPLETED (18:30, 18:48); tests, capture and whitening controls passed (energy 3.154 / 3.893,
  ranks o / up / down 408 / 726 / 789 and 350 / 673 / 739). Mem / Util / Priv / EM / FQ / Overall: rho 60 0.9023 /
  0.8552 / 0.8955 / 0.0638 / -1.8995 / 0.8838; rho 300 0.9437 / 0.8352 / 0.8825 / 0.0719 / -1.5250 / 0.8849.
- [x] 01:33 ReCAST C down_up_o rho 30 seeds 1 / 2: array 24995918 (otherwise as 24994488_1, 0.8932).
  Result (01:56): both COMPLETED (17:20, 19:20); controls passed (bases identical to seed 0). Seed 1 0.9023 / 0.8806 /
  0.8777 / 0.0606 / -2.0999 / 0.8867; seed 2 0.8977 / 0.8835 / 0.8730 / 0.0746 / -2.6394 / 0.8846.
- [x] 01:33 ReCAST C mlp_o rho 10 / 30 / 100: array 24995919 (seed 0; restricted_semu free_all - for mlp_o the norms
  and q/k/v - lr 1e-5 constant, nll). No mlp_o reference in this configuration yet.
  Result (01:58): all COMPLETED (16-21 min); controls passed (plain energy per covector 3.965; whitened 1.877 /
  2.483 / 3.053; gate_proj ranks plain 694, whitened 849 / 804 / 764). rho 10 0.9139 / 0.8427 / 0.8003 / 0.0010 /
  -2.0999 / 0.8498; rho 30 0.9308 / 0.8277 / 0.7734 / 0.0129 / -1.6153 / 0.8390; rho 100 0.9436 / 0.8150 / 0.8184 /
  0.0011 / -1.8026 / 0.8550.
- [x] 01:36 Decision (user): "tak, pusc" - plain-basis references: mlp_o seed 0, and down_up_o seeds 1 / 2.
- [x] 01:36 ReCAST plain-basis reference mlp_o: array 24995987 (filter none, seed 0; otherwise as 24995919).
  Result (01:55): COMPLETED in 16:10; controls passed. 0.9470 / 0.8049 / 0.9037 / 0.0011 / -2.7539 / 0.8811. C lowers
  mlp_o at every rho tried (0.8498 / 0.8390 / 0.8550): Utility +0.01-0.04, Privacy -0.09 to -0.13.
- [x] 01:36 ReCAST plain-basis reference down_up_o seeds 1 / 2: array 24995988 (otherwise as 24991498_0, 0.8674).
  Result (01:57): both COMPLETED (18:33, 15:22); controls passed. Seed 1 0.9152 / 0.8382 / 0.8474 / 0.0840 / -2.5271 /
  0.8656; seed 2 0.8868 / 0.8536 / 0.8537 / 0.0865 / -3.2341 / 0.8644.
- [x] 12:41 Code: `collect_recast_tofu.py` fills arguments that older runs lack with their present defaults
  (`with_defaults`: retain_loss nll, target_blocks all; restricted: no filter, no whitening, no layer scaling,
  full_space; restricted_semu: phase2_free none) and treats an argument set to null as absent, so equal
  configurations group together (24991498_0 had split from 24995988 over `whitening_rho: null`). Gate: `py_compile`.
  Earlier summaries are superseded where groups merged.
- [x] 12:41 Aggregation (job 25007837, `runtime_artifacts/ai-engram/recast_tofu/summary_20261023_1239.json`), down_up_o,
  restricted_semu free_all, lr 1e-5 constant, nll, seeds 0-2 (n = 3), mean +- sample std:
  - plain basis: Mem 0.9052 +- 0.0159, Util 0.8470 +- 0.0079, Priv 0.8480 +- 0.0054, EM 0.0855 +- 0.0013, FQ -2.7261 +-
    0.4434, Overall 0.8658 +- 0.0015;
  - C rho 30: Mem 0.9039 +- 0.0070, Util 0.8852 +- 0.0057, Priv 0.8760 +- 0.0026, EM 0.0699 +- 0.0080, FQ -2.3142 +-
    0.2863, Overall 0.8882 +- 0.0045;
  - C rho 10: Overall 0.8794 +- 0.0064 (24994962).
  C at rho 30 beats the plain basis by +0.0224 Overall (Utility +0.038, Privacy +0.028, EM and FQ better, Mem
  equal); each seed of C is above every seed of the plain basis.
- [x] 13:22 Decision (user): C with "shrinkiem 0.2, oraz evr=0.98 - kombinacje".
- [x] 13:22 ReCAST C rho 30, s_min x EVR: array 25009744 (tasks 0-3 = s_min 0.2 / EVR 0.97, 0.2 / 0.98, 0.3 / 0.97, 0.3 /
  0.98), down_up_o, seed 0; otherwise as 24994488_1. Task 2 repeats 24994488_1 (0.8932): reproduction control.
  Result (13:51): tasks 0, 1, 3 COMPLETED (15:26, 16:00, 18:47; task 2 cancelled); tests, capture and whitening
  controls passed (EVR 0.98: energy 2.470 whitened / 3.892 plain, median ranks o / up / down 545 / 943 / 1117 vs 448 /
  788 / 957 at 0.97). Mem / Util / Priv / EM / FQ / Overall: s_min 0.2 EVR 0.97 0.9615 / 0.7903 / 0.8025 / 0.0105 /
  -0.9541 / 0.8447; 0.2 / 0.98 0.9728 / 0.7078 / 0.8060 / 0.0192 / -0.4919 / 0.8149; 0.3 / 0.98 0.9407 / 0.8205 /
  0.8721 / 0.0248 / -1.5250 / 0.8750 (0.3 / 0.97: 0.8932).
- [x] 13:26 Cancelled 25009744_2 (s_min 0.3 / EVR 0.97, a repeat of 24994488_1): redundant, reproducibility of seed-0 runs is already shown by four exact repeats (24987066_0/5, 24989313_0/3, 24990942_2/8).
- [x] 13:27 Decision (user): "dolicz do tych nowych kombinacji tez zwykly recast" - plain basis at the three new s_min / EVR
  combinations (0.3 / 0.97 exists: 24991498_0).
- [x] 13:27 ReCAST plain basis s_min x EVR: arrays 25009891 (s_min 0.2, EVR 0.97 / 0.98) and 25009893 (s_min 0.3, EVR 0.98);
  down_up_o, restricted_semu free_all, lr 1e-5 constant, nll, seed 0.
  Result (13:55): all COMPLETED (12:24, 14:55, 15:08); controls passed (EVR 0.98 median ranks 402 / 794 / 942). s_min
  0.2 EVR 0.97 0.9343 / 0.7921 / 0.9300 / 0.0409 / -1.7079 / 0.8803; 0.2 / 0.98 0.9538 / 0.6981 / 0.8253 / 0.0144 /
  -0.6125 / 0.8124; 0.3 / 0.98 0.9478 / 0.8082 / 0.9250 / 0.0387 / -1.4369 / 0.8893 (0.3 / 0.97: 0.8674).
  Finding: at s_min 0.3 the plain basis gains from a larger subspace (EVR 0.98: 0.8893, ranks 402 / 794 / 942) and
  then matches C at EVR 0.97 (0.8932, median ranks 448 / 788 / 957) with a different profile (Util 0.808 vs 0.892, Priv
  0.925 vs 0.877). C enlarges the subspace at a fixed EVR, so part of its gain over the plain basis at EVR 0.97 may
  come from the rank, not from the whitening; a matched-rank comparison is open (seed 0 only).
- [x] 14:11 Decision (user): "ok, rob to co proponujesz" - plain basis at s_min 0.3 / EVR 0.98, seeds 1 / 2, for a near
  rank-matched comparison with C rho 30 at EVR 0.97 (n = 3 each).
- [x] 14:11 ReCAST plain basis s_min 0.3 / EVR 0.98 seeds 1 / 2: array 25011397 (otherwise as 25009893_0, 0.8893).
  Result (14:31): both COMPLETED (17:47, 17:41); 26 parity tests passed, frozen tensors moved 0, capture control
  passed. Mem / Util / Priv / EM / FQ / Overall: seed 1 0.9109 / 0.8130 / 0.9173 / 0.0396 / -1.9986 / 0.8777; seed 2
  0.9201 / 0.8091 / 0.9096 / 0.0466 / -1.8995 / 0.8766.
- [x] 14:31 Aggregation (job 25012475, `runtime_artifacts/ai-engram/recast_tofu/summary_20261023_1431.json`), down_up_o,
  restricted_semu free_all, lr 1e-5 constant, nll, s_min 0.3, seeds 0-2 (n = 3), mean +- sample std:
  - plain basis EVR 0.97 (median ranks 311 / 645 / 782): Overall 0.8658 +- 0.0015;
  - plain basis EVR 0.98 (402 / 794 / 942): Mem 0.9263 +- 0.0192, Util 0.8101 +- 0.0026, Priv 0.9173 +- 0.0077, EM
    0.0416 +- 0.0043, FQ -1.7783 +- 0.2998, Overall 0.8812 +- 0.0070;
  - C rho 30 EVR 0.97 (448 / 788 / 957): Mem 0.9039 +- 0.0070, Util 0.8852 +- 0.0057, Priv 0.8760 +- 0.0026, EM
    0.0699 +- 0.0080, FQ -2.3142 +- 0.2863, Overall 0.8882 +- 0.0045.
  Finding: at near-matched rank C leads the plain basis by +0.007 Overall, about one standard deviation (not
  established with n = 3); most of C's +0.022 over the plain basis at EVR 0.97 comes from the larger subspace. At
  matched rank the whitening shifts the profile: Utility +0.075, Privacy -0.041, Mem -0.022, EM 0.070 vs 0.042,
  FQ -2.31 vs -1.78.
- [x] 14:43 Conclusion (written to the doc "ReCAST retain-aware variants A and C (TOFU)" on user request): the edit
  budget (rank, shrink strength) moves Overall more than the choice of directions; at near-matched rank C beats the
  plain basis by +0.007 (about one std, n = 3) and mainly shifts the profile towards Utility; A never beats the
  plain basis; retain-only phase two does not repair pre-TOFU knowledge. Design notes in the doc: set the budget
  explicitly and compare methods at equal budget, pick directions by a forget-vs-retain contrast, protect general
  knowledge in the edit itself, keep phase two conservative (constant 1e-5, no KL to Init), tune per profile for
  the harmonic mean, decide on n >= 3.
- [x] 15:23 Decision (user): "sprawdz hipoteze ... o jawnie wiekszym budzecie rangi".
- [x] 15:23 Code: explicit edit budget. `recast_core.svd_basis(..., rank=)` and `whitened_svd_basis(..., rank=)` take a fixed
  rank (evr then reports the energy kept); `warm_start(rank_multiplier=m)` gives every restricted target rank
  round(m * r0), r0 the plain-basis rank at the EVR threshold, for the plain and the whitened basis alike (report
  `reference_rank`); test added. `recast_tofu.py --restriction_rank_multiplier` (restricted only); with a budget the
  whitening control compares against the plain basis at the same rank. Launcher axis `RANK_MULTS` (run-name suffix
  `_rankx<m>`); collector shows it. SHA-256 prefixes recast_core, recast_tofu: 533879de a0b7f4a4 . Gate: `py_compile`, `bash -n`,
  dry-run of the arguments.
- [x] 15:23 ReCAST explicit budget, down_up_o, s_min 0.3, EVR 0.97 reference ranks, seed 0, restricted_semu free_all,
  lr 1e-5 constant, nll: plain basis m 1.25 / 1.5 / 2 (array 25014344; m 1 = 24991498_0, 0.8674) and C rho 30 m 1 /
  1.25 / 1.5 / 2 (array 25014345).
  Result (15:51): all COMPLETED (13-19 min); 27 / 29 parity tests passed, frozen tensors moved 0, capture and
  whitening controls passed; median ranks o / up / down equal for both bases at each m (1: 311 / 645 / 782; 1.25:
  389 / 806 / 978; 1.5: 466 / 968 / 1173; 2: 622 / 1290 / 1564). Mem / Util / Priv / EM / FQ / Overall; ROUGE real
  authors / world facts:
  - plain m 1.25 0.9318 / 0.7906 / 0.8997 / 0.0222 / -1.3509 / 0.8697, 0.164 / 0.419; m 1.5 0.9675 / 0.7474 / 0.8913 /
    0.0064 / -1.0291 / 0.8587, 0.124 / 0.382; m 2 0.9386 / 0.4293 / 0.7767 / 0.0690 / -0.1549 / 0.6407, 0.028 / 0.238;
  - C m 1 0.8621 / 0.8914 / 0.8062 / 0.1303 / -3.7497 / 0.8517, 0.297 / 0.539; m 1.25 0.9255 / 0.8868 / 0.8811 / 0.0694 /
    -1.5250 / 0.8974, 0.360 / 0.582; m 1.5 0.9584 / 0.8216 / 0.8960 / 0.0314 / -1.9986 / 0.8885, 0.181 / 0.493; m 2
    0.9478 / 0.5578 / 0.7756 / 0.0489 / -0.3297 / 0.7251, 0.050 / 0.281.
  Finding (seed 0): at equal rank C loses at m 1 (-0.016: less forgetting) and wins from m 1.25 on (+0.028, +0.030,
  +0.084), mainly through Utility (+0.096 / +0.075 / +0.129) at nearly the same Memorization and Privacy: the
  contrast-chosen directions take a larger edit budget with less collateral damage. Best: C m 1.25, 0.8974 (best
  single run of the campaign). Plain stays at 0.86-0.87 up to m 1.5 and breaks at m 2.
- [x] 15:35 Note: tasks 25014345_1-3 started while the code below was being written and loaded intermediate versions
  (recast_core 72ae94ca for 1-3, recast_tofu c9c17cad for 3; parity tests 27 / 29 passed); the paths they use (plain
  and whitened bases with a budget) are unchanged by those edits. Per-run SHA-256 is in each results.json.
- [x] 15:35 Decision (user): basis D from the forget and retain representations - "przygotuj najlepiej kazdy wariant" of
  the output side.
- [x] 15:35 Deviation / Code: basis D (`recast_core.representation_basis`): input covectors = top-r solutions of C_f v =
  gamma (C_r + eps I) v (answer-token input covariances of forget10 and retain90 at Init, eps = rho tr(C_r) / d),
  orthonormalised; output side identity (one-sided restriction, `u=None`: `project`, `SemuAdapterLinear` and
  `adapter_core_and_anchor` support it), gradient (left singular vectors of G_f V) or representation (the same
  contrast for W C W^T); rank = explicit budget; `evr` = forget-gradient energy kept. Tests: one-sided projection
  and adapter (start at W*, rows stay in span(Vh)); a diagonal toy where forget favours e1 and retain e0 picks e1 on
  every output side, orthonormal U, U along G_f v. `recast_tofu.py --restriction_representation` (optional; needs
  the budget and --whitening_rho, excludes whitening); forget covariance pass; positive control: the D covectors
  have a higher forget/retain input-energy ratio than plain covectors of the same rank, else stop; `[repr]`
  lines. Launcher FILTERS entry repr/rho/side; collector shows repr. SHA-256 prefixes recast_core, recast_tofu: 72ae94ca fa7a2587 .
  Gate: `py_compile`, `bash -n`, dry-run of the arguments.
- [x] 15:35 ReCAST basis D: array 25015009 (tasks 0-2 = output side identity / gradient / representation), rho 30, budget m 1
  (plain ranks at EVR 0.97), down_up_o, s_min 0.3, seed 0, restricted_semu free_all, lr 1e-5 constant, nll.
  Result (15:59): all COMPLETED (14:44, 14:28, 20:13); 29 parity tests passed, frozen tensors moved 0, capture control
  passed; SHA-256 recast_core 72ae94ca, recast_tofu fa7a2587 in every task; representation control passed (forget /
  retain input-energy ratio of the basis 1.222 vs plain 1.081; forget-gradient energy kept identity and gradient
  0.661, representation 0.122, plain 0.971). Warm-start relative weight change o / up / down: identity 0.223 / 0.377
  / 0.192, gradient 0.085 / 0.109 / 0.114, representation 0.214 / 0.363 / 0.184 (plain m 1: 0.090 / 0.111 / 0.114);
  trainable adapter cores: identity 113,680,384, the others 15,673,087. Mem / Util / Priv / EM / FQ / Overall; ROUGE
  real authors / world facts: identity 0.8963 / 0.3624 / 0.5252 / 0.2564 / -0.3811 / 0.5191, 0.022 / 0.178; gradient
  0.9028 / 0.8821 / 0.8404 / 0.0882 / -2.2034 / 0.8743, 0.256 / 0.618; representation 0.8891 / 0.4650 / 0.4715 / 0.2666 /
  -0.8108 / 0.5559, 0.035 / 0.205.
  Finding (seed 0): at m 1, D with the gradient output side is the best basis (0.8743; plain 0.8674, C 0.8517), with
  Utility near C (0.882) and Memorization / Privacy near plain. The identity and representation output sides
  remove 2-3x more of W at the same rank (output directions of high energy) and break Utility: the rank alone is
  not the edit budget, the weight change is.
- [x] 16:05 Decision (user): seeds for the best candidates at budget m 1.25.
- [x] 16:05 ReCAST budget m 1.25 seeds: array 25016368 (plain and C rho 30, seeds 1 / 2; seed 0 = 25014344_0, 25014345_1) and
  array 25016369 (D rho 30 gradient output side, seeds 0 / 1 / 2); down_up_o, s_min 0.3, EVR 0.97 reference ranks,
  restricted_semu free_all, lr 1e-5 constant, nll; code unchanged since 25015009 (72ae94ca / fa7a2587).
  Result (2026-10-23 16:31): all COMPLETED (13-21 min); 29 parity tests passed, frozen tensors moved 0, capture,
  whitening and representation controls passed; every task loaded 72ae94ca / fa7a2587. Warm-start relative weight
  change o / up / down: plain 0.112 / 0.137 / 0.142, C 0.113 / 0.137 / 0.142, D 0.107 / 0.136 / 0.142 (equal budget).
  Aggregation (job 25017189, `runtime_artifacts/ai-engram/recast_tofu/summary_20261023_1631.json`), n = 3, mean +-
  sample std:
  - plain: Mem 0.9440 +- 0.0110, Util 0.8039 +- 0.0116, Priv 0.8973 +- 0.0371, EM 0.0328 +- 0.0096, FQ -1.2143 +-
    0.1248, Overall 0.8775 +- 0.0151 (0.8697, 0.8949, 0.8680);
  - C rho 30: Mem 0.9061 +- 0.0185, Util 0.8785 +- 0.0080, Priv 0.8803 +- 0.0034, EM 0.0740 +- 0.0051, FQ -2.4287 +-
    0.8588, Overall 0.8881 +- 0.0081 (0.8974, 0.8833, 0.8836);
  - D rho 30, gradient output side: Mem 0.9506 +- 0.0070, Util 0.8647 +- 0.0104, Priv 0.9361 +- 0.0086, EM 0.0144 +-
    0.0010, FQ -0.8821 +- 0.0717, Overall 0.9155 +- 0.0061 (0.9225, 0.9110, 0.9131).
  Finding: at an equal budget D beats C by +0.027 and the plain basis by +0.038 Overall, every D seed above every
  other seed; D keeps the forgetting and Privacy of the plain basis (or better: EM 0.014, FQ -0.88) with Utility
  close to C. Best configuration of the campaign: D, m 1.25, Overall 0.9155 +- 0.0061.
- [x] 16:14 Decision (user): implement the two C + D combinations.
- [x] 16:14 Deviation / Code (written in a staging copy while 25016368/9 were pending, copied into the repo after all seven
  tasks had loaded 72ae94ca / fa7a2587): `representation_basis(input_mix=beta)` - input-side forget signal
  N = (1 - beta) G_f^T G_f / tr + beta C_f / tr against C_r + eps I (beta 1 = D, unchanged; beta 0 = the input side of
  C), and output side `whitened_gradient` (top-r left singular vectors of G_f (C_r + eps I)^-1/2, the output side of
  C). Joint version = beta 0.5 with the gradient output side; sequential version = D input (beta 1) with the
  whitened-gradient output side. Test: beta 0 with the whitened-gradient output side spans exactly the C basis of
  the same rank, beta 0 with the gradient output side has C's output span, beta 1 is invariant to scaling C_f.
  `recast_tofu.py --representation_mix` (optional), `--restriction_representation whitened_gradient`; launcher
  repr entry repr/rho/side[/beta]; collector shows the mix. SHA-256 prefixes recast_core, recast_tofu, tests: 6088f6de e9a95658 fc2cb5cd .
  Gate: `py_compile`, `bash -n`, dry-run (joint and sequential at m 1.25, seed 0, ready, not submitted).

## 2026-10-25

- [x] 20:35 Decision (user): run the two C + D combinations and D at m 1.5; describe D step by step against C in the doc.
- [x] 20:35 $HOME quota at 96.7 % blocked submission (hook); with the user's consent moved
  ~/.vscode-server/data/CachedExtensionVSIXs (2.1 GB, a download cache) to
  /net/storage/pr3/plgrid/plggdnnp/archive/plgkrzepk/vscode-server-CachedExtensionVSIXs-20261025 (copy verified with diff
  before removal); two .nfs files held open by a running process stay until it closes them; quota 87.9 %.
- [ ] 20:35 ReCAST C + D combinations: array 25084397 (task 0 joint, beta 0.5, gradient output side; task 1 sequential, D input,
  whitened-gradient output side) and D at m 1.5: array 25084398; rho 30, down_up_o, s_min 0.3, seed 0, restricted_semu
  free_all, lr 1e-5 constant, nll; code 6088f6de e9a95658 fc2cb5cd (recast_core, recast_tofu, tests).
  Result (21:14): all COMPLETED (18:15, 17:41, 18:12); 30 parity tests passed, frozen tensors moved 0, capture and
  representation controls passed (forget-gradient energy kept: joint 0.859, sequential 0.692, D m 1.5 0.726; plain
  0.981 / 0.988); no write errors although $HOME was near its quota. Warm-start weight change o / up / down: joint and
  sequential 0.108 / 0.136 / 0.142, D m 1.5 0.13 / 0.163 / 0.170. Mem / Util / Priv / EM / FQ / Overall; ROUGE real
  authors / world facts: joint 0.9484 / 0.8573 / 0.9351 / 0.0270 / -0.6125 / 0.9118, 0.248 / 0.480; sequential 0.9641 /
  0.8444 / 0.8894 / 0.0211 / -0.5510 / 0.8966, 0.213 / 0.467; D m 1.5 0.9528 / 0.8155 / 0.8205 / 0.0449 / -0.2357 /
  0.8585, 0.180 / 0.454. Neither combination beats D at m 1.25 (0.9155 +- 0.0061; seed 0 0.9225); m 1.5 is too
  large for D (seed 0).
- [x] 21:35 $HOME back at 98.2 % / 95.5 % files: the openai.chatgpt 26.1021 update installed (1.2 GB, 22.7k files)
  while the running server still holds the obsolete 26.1007 (990 MB) and four .nfs downloads (1.7 GB); these free on
  a VS Code server restart (the user's action). Submission is blocked until then.
- [x] 21:35 Code: `replication/scripts/basis_energy_analysis.py` and `replication/scripts/bash/recast_tofu/basis_energy_gh200.sh`
  (no training): for the plain, C and D bases at the m 1.25 budget, per target the mean forget / retain input energy per
  covector (relative to an average direction) and the warm-start change of the layer outputs on forget / retain answer
  tokens, tr(D C D^T) / tr(W C W^T), their ratio and ||D|| / ||W||; positive control: equal ranks of the three bases.
  Tests the hypothesis that C favours directions where the forget activations are weak. Gate: `py_compile`, `bash -n`;
  not submitted (quota).
- [x] 20:51 $HOME at 99.4 % space / 97.6 % files (the openai.chatgpt extension keeps re-downloading a 428 MB update it
  cannot install; each failed attempt leaves an open, deleted .nfs file). With the user's consent removed three unused
  VS Code items: the CLI server build cli/servers/Stable-2d0b7506 (645 MB, 4,536 files, no process), the obsolete
  extension anthropic.claude-code-2.1.306 (291 MB; marked obsolete by VS Code; its native binary stays as an open
  .nfs file until the process holding it exits) and the uninstalled openai.chatgpt-26.1021 downloads in
  CachedExtensionVSIXs and its .trash. Quota 82.8 % space / 72.8 % files. Still held: three .nfs downloads (1.28 GB)
  and the 2.1.306 binary (290 MB), released when the VS Code server restarts.
- [x] 22:41 After a window reload the main VS Code server process (running since 2026-10-22) still held the four .nfs
  downloads (1.7 GB) and eight hung ugrep processes from 2026-10-22 held the 2.1.306 binary (290 MB). With the
  user's consent removed the obsolete openai.chatgpt-26.1007 (990 MB, 19.7k files, marked obsolete, no process) and
  the cached openai.chatgpt-26.1021 package (428 MB, the version is installed). Quota 84.4 % space / 75.9 % files.
- [x] 22:41 Basis energy analysis: job 25086218 (down_up_o, m 1.25, rho 30, s_min 0.3, EVR 0.97 reference ranks; SHA-256
  basis_energy_analysis, recast_core: 347acdce 6088f6de ).
  Result (22:46): COMPLETED in 4:53; control passed (equal ranks of the three bases at every target); answer tokens
  retain 124,617, forget 14,688; 48 targets. Means over targets, e_f / e_r (energy per covector relative to an average
  direction) | q_f / q_r (warm-start output change on forget / retain answer tokens) | selectivity q_f / q_r | weight
  change: plain 4.228 / 3.893 | 0.0806 / 0.0747 | 1.063 | 0.131; C 3.265 / 2.849 | 0.0613 / 0.0538 | 1.122 | 0.131;
  D 5.952 / 4.224 | 0.0821 / 0.0693 | 1.201 | 0.129. down_proj: plain 5.911 / 5.243 | 0.1189 / 0.1066 | 1.094; C 3.907 /
  3.120 | 0.0836 / 0.0688 | 1.199; D 10.843 / 6.311 | 0.1318 / 0.1034 | 1.405. o_proj and up_proj show the same order
  with smaller gaps (selectivity o 1.038 / 1.079 / 1.092, up 1.056 / 1.087 / 1.106).
  Finding: the hypothesis holds. C's covectors carry less retain energy than the plain ones (2.85 vs 3.89, still
  above an average direction) and as much less forget energy (3.26 vs 4.23), so its shrink changes forget outputs
  24 % less (and retain outputs 28 % less); D's covectors carry 41 % more forget energy than the plain ones at
  similar retain energy, so its shrink changes forget outputs as much as the plain basis and retain outputs 7 %
  less: the highest selectivity (1.20). The orders match the trained runs: q_r C < D < plain as Utility C > D >
  plain, q_f D >= plain > C as Memorization D >= plain > C.

## 2026-10-26

- [x] 20:24 Decision (user): analyse the selectivity of the bases on every matrix type to choose what to shrink; explain
  q_f; training-free tuning is of interest; budget-by-selectivity and soft shrink come after these results.
- [x] 20:24 Basis energy analysis on every decoder projection (TARGETS=decoder_linear: q, k, v, o, gate, up, down): job 25222020,
  otherwise as 25086218 (launcher now takes TARGETS; output basis_energy_<profile>_<job>.json).
  Result (20:31): COMPLETED in 6:34; control passed (equal ranks at all 112 targets). Selectivity q_f / q_r (mean over
  the 16 blocks) plain / C / D and D's q_f: q_proj 1.003 / 1.075 / 1.076, 0.015; k_proj 1.008 / 1.070 / 1.083, 0.030;
  v_proj 1.045 / 1.092 / 1.095, 0.095; o_proj 1.038 / 1.079 / 1.092, 0.069; gate_proj 1.037 / 1.074 / 1.083, 0.033;
  up_proj 1.056 / 1.087 / 1.106, 0.046; down_proj 1.094 / 1.199 / 1.405, 0.132. Per block, D on down_proj 1.22-1.33
  (block 1: rank-1 basis, q ~ 0), every other type 1.02-1.24 (mostly 1.05-1.13).
  Finding: down_proj is the only matrix type whose shrink is clearly selective, and it carries the largest forget
  output change; q_proj and k_proj with the plain basis are not selective at all (1.003 / 1.008), consistent with the
  poor down_up_q run (0.7048); gate_proj is weak (1.08), consistent with gate costing Utility.
- [x] 20:26 Code: `replication/scripts/warmstart_proxy.py` and `replication/scripts/bash/recast_tofu/warmstart_proxy_gh200.sh`:
  for 37 down_up_o configurations that have trained runs (plain EVR / s_min / budgets, C over rho / budgets / s_min /
  EVR, D variants and budgets, A over kappa) the warm start alone on copies of the Init weights, the output change on
  forget / retain answer tokens per target (q = tr(D C D^T) / tr(W C W^T)), aggregated as means and energy-weighted
  totals Q_f, Q_r, selectivity Q_f / Q_r; joined with a fresh collector summary of the trained runs (phase two
  restricted_semu free_all, lr 1e-5 constant, nll) and Spearman correlations with Mem / Util / Priv / Overall.
  SHA-256 prefixes warmstart_proxy, recast_core, collect_recast_tofu: c75fd5b4 6088f6de b641669b . Gate: `py_compile`, `bash -n`.
- [x] 20:26 Warm-start proxy check: job 25222073.
  Result (20:47): COMPLETED in 20:33; 37 of 37 configurations joined with their trained runs (output
  `runtime_artifacts/ai-engram/recast_tofu/warmstart_proxy_25222073.json`). Spearman rho (p) across configurations,
  proxy vs trained Memorization / Utility / Privacy / Overall: Q_f +0.79 (0.000) / -0.97 (0.000) / +0.40 (0.013) /
  +0.20 (0.224); Q_r +0.80 / -0.97 / +0.40 / +0.20; selectivity Q_f / Q_r -0.61 / +0.65 / -0.57 / -0.39 (0.018);
  mean-over-targets versions alike; relative weight change +0.35 / -0.53 / -0.05 / -0.13.
  Finding: Q_f and Q_r move together (selectivity 1.03-1.15 except the tiny-edit A / C runs), so they measure one
  edit magnitude Q; Q predicts Utility almost exactly (falls with Q) and Memorization well (rises with Q). Overall is
  an inverted U in Q (Q_f ~ 0 at kappa 0.1: 0.108; 0.03-0.045: 0.85-0.92; 0.075: 0.641; 0.35: 0.519), hence no
  monotone correlation; the negative selectivity correlation comes from the tiny edits. At a matched Q_f near 0.04
  the order follows selectivity: plain 1.044 (0.878-0.881), C 1.074-1.085 (0.875-0.889), D 1.107-1.112 (0.897-0.916).
  Training-free rule suggested: set the budget to Q_f ~ 0.035-0.045 and pick the basis / allocation with the highest
  selectivity at that Q_f (to be confirmed with a fit and a held-out prediction).
- [x] 23:20 Decisions (user): trained results for profiles with v_proj (down + up + v instead of o, and down + up + o + v);
  selectivity of lm_head with the head tied to the input embedding and untied.
- [x] 23:20 Code: profiles `down_up_v`, `down_up_o_v` and `head` (lm_head only) in TARGET_PROFILES, test added;
  `basis_energy_analysis.py --untie_lm_head`, tr(D C D^T) computed as sum((D C) * D) (no out x out matrix for the
  128k-row head), and for a tied head the embedding-row change sum_t p(t)||D_t||^2 / sum_t p(t)||W_t||^2 over the
  non-padding input-token frequencies of forget / retain; launcher env UNTIE. SHA-256 prefixes recast_core,
  basis_energy_analysis, tests: 319a9190 fad3d40e f0cfab7c . Gate: `py_compile`, `bash -n`.
- [x] 23:20 lm_head analysis: jobs 25231798 (tied) and 25231799 (untied), plain / C / D at m 1.25, rho 30.
  Result (23:24): both COMPLETED (3:39, 3:24); control passed. Head output selectivity plain / C / D 1.062 / 1.091 /
  1.110 untied, 1.058 / 1.091 / 1.105 tied; head output change q_f 0.0015 / 0.0012 / 0.0019 untied, 0.0031 / 0.0023 /
  0.0037 tied; weight change ~2.7 % / ~3.7 %. Tied only: embedding rows read by the inputs change q_emb_f / q_emb_r
  0.1807 / 0.1563 plain, 0.1716 / 0.1474 C, 0.1797 / 0.1567 D (selectivity 1.15-1.16). Finding: the head is no more
  selective than o / up, and a tied head's shrink rewrites ~15 % of the embedding energy every retain input reads.
- [x] 23:20 ReCAST D (rho 30, gradient output side, m 1.25) on down_up_v and down_up_o_v: array 25231800, seed 0, s_min 0.3,
  EVR 0.97 reference ranks, restricted_semu free_all, lr 1e-5 constant, nll.
  Result (23:40): both COMPLETED (19:39, 15:35); 31 parity tests passed, frozen tensors moved 0, capture and
  representation controls passed (ratio 1.203 / 1.180 vs plain 1.072 / 1.069). Mem / Util / Priv / EM / FQ / Overall;
  ROUGE real authors / world facts: down_up_v 0.9582 / 0.8643 / 0.9208 / 0.0246 / -0.9541 / 0.9128, 0.263 / 0.550;
  down_up_o_v 0.9544 / 0.8009 / 0.8306 / 0.0406 / -0.1549 / 0.8570, 0.178 / 0.434 (down_up_o, n = 3: 0.9155 +- 0.0061).
- [x] 23:20 Training-free prediction for the same profiles (and down_up_o as a check): job 25231801 (warmstart_proxy.py, D m
  1.25 and plain EVR 0.97 per profile; no trained summary - the prediction is made before the runs finish).
  Result (23:32): COMPLETED in 11:20. D m 1.25 Q_f / Q_r / selectivity / mean q_f / mean q_r / mean selectivity:
  down_up_v 0.0443 / 0.0398 / 1.112 / 0.0910 / 0.0773 / 1.202; down_up_o_v 0.0444 / 0.0399 / 1.111 / 0.0854 / 0.0737 /
  1.174; down_up_o 0.0407 / 0.0366 / 1.112 / 0.0821 / 0.0693 / 1.201 (equal to 25222073). Prediction made (chat, before
  the results): both v profiles close to down_up_o. Outcome: right for down_up_v (0.9128), wrong for down_up_o_v
  (0.8570). The energy-weighted Q normalises by the total output energy, which down_proj dominates, so an extra
  edited matrix barely moves it; the sum over matrices of q_r (down_up_o 3.33, down_up_v 3.71, down_up_o_v 4.72) and
  the mean selectivity (1.201 / 1.202 / 1.174) do separate it. Hypothesis: damage accumulates over edited matrices,
  so profiles with different matrix counts need a summed proxy.
- [x] 00:06 (2026-10-27) Decisions (user): compare the metrics right after the warm start with those after phase two
  (how much the fine-tuning gives); correlations of the training-free proxies with EM and FQ; carry out the two
  proposals: (1) the edit budget in forget-output units, per target the rank that reaches a target q_f, so the
  energy-weighted total Q_f reaches it too; (2) allocation by selectivity, training-free first: the budget on
  down_proj only (larger m) against down + up + o_proj, then train the best 2-3.
- [x] 00:06 Code: `recast_tofu.py --epochs 0` evaluates the warm-started model without phase two (row suffix "warm start
  only", `phase_two.skipped`); `recast_core.warm_start(forget_output_target=q)` with `basis_prefix` /
  `prefix_for_forget_output`: each restricted target's basis family is built at full rank, and the smallest prefix
  rank whose delta has q_f = tr(D C_f D^T) / tr(W C_f W^T) >= q is found by bisection (D's gradient output side is
  recomputed per prefix as the QR of G_f V); `--restriction_forget_output_target` with a positive control (every
  active target reaches q or uses its full rank); launcher budget axis `qf/<q>` and `EPOCHS` from --export (default
  10). `warmstart_proxy.py`: functional proxies (mean answer-token NLL increase of the warm-started model on forget,
  retain, real authors, world facts; dL_f / dL_r, dL_f / dL_world) with a restore control, config keys `qf` and
  `profile` (`--collect_targets` covers all profiles), EM and FQ among the correlated metrics. Collector shows the
  forget-output target. Two tests added (prefix bases equal the direct bases for plain / C / D; the budget picks the
  smallest rank reaching the target). SHA-256 prefixes recast_core, recast_tofu, warmstart_proxy,
  collect_recast_tofu, tests: 16167048 55116943 2824cc47 836e6933 2b523b32 . Gate: `py_compile`, `bash -n`; the
  parity tests run in each recast array task.
- [x] 00:07 Warm start only (EPOCHS=0), down_up_o, seed 0, s_min 0.3, EVR 0.97, restricted_semu: array 25232879 (plain and C
  rho 30, each at the EVR rank and at m 1.25) and array 25232883 (D rho 30 gradient output side at m 1, 1.25, 1.5).
  Result (00:30): all 7 COMPLETED (13:05-21:54); 33 parity tests passed in each, adapter start error <= 1.4e-8, capture
  control passed. Every configuration scores Overall 0.0000 (Mem 0, Util 0-0.002, Priv 0, rescaled EM clipped at
  1, FQ -7.8 to -20.5). Raw: forget EM 0.007-0.047 (Init Fine. 0.974, Retain 0.586), forget answer probability
  2e-5 to 1.2e-4 (0.881 / 0.116), Model Utility 0-0.0007 (0.599 / 0.592), forget gibberish score 0.008-0.125 (0.86 /
  0.905), real-author / world-fact ROUGE <= 0.034; generations are token noise on every split. Plain EVR / m 1.25:
  FQ -20.5 / -19.9; C rho 30 EVR / m 1.25: -10.9 / -10.7; D m 1 / 1.25 / 1.5: -7.8 / -12.7 / -14.5. The proxy job
  (25232884) confirms it on NLL: the warm start raises the answer NLL from 0.14 to ~10 nats on forget and retain
  alike (plain EVR: dL_f 9.99, dL_r 9.99; ln 128256 = 11.76 is a uniform guess).
  Finding: the s_min 0.3 warm start does not remove the forget knowledge selectively; it destroys the model (forget,
  retain and pre-TOFU knowledge, and fluency). Phase two rebuilds Utility (0.80-0.89) and Privacy (0.85-0.94) from
  retain data, and Memorization stays high (0.90-0.95) because the forget facts are not relearned. The bases differ
  in what phase two can restore, not in the damage right after the warm start, which is total for all seven.
- [x] 00:07 Training-free proxies with functional proxies and EM / FQ correlations: job 25232884, the 37 configurations of
  25222073 plus D m 1.25 on down_up_v and down_up_o_v (trained), and untrained screens: D with qf 0.02-0.08, plain and
  C rho 30 at qf 0.04 (down_up_o); D on down_proj only at m 1.25-4 and qf 0.03-0.1; D m 1.25 on down + up.
  Result (01:11): COMPLETED in 1:03:51; NLL restore control passed; 39 of 55 configurations joined (output
  `runtime_artifacts/ai-engram/recast_tofu/warmstart_proxy_25232884.json`). Init Fine. answer NLL forget 0.139,
  retain 0.133, real authors 4.744, world facts 6.710. Spearman rho (p), proxy vs trained Mem / Util / Priv / EM / FQ
  / Overall: Q_f +0.77 / -0.95 / +0.38 (0.017) / -0.65 / +0.93 / +0.20 (0.229); mean q_f +0.80 / -0.94 / +0.34 /
  -0.68 / +0.96 / +0.19; selectivity -0.54 / +0.62 / -0.52 / +0.58 / -0.41 (0.010) / -0.36 (0.026); dL_f +0.70 /
  -0.88 / +0.40 / -0.60 / +0.86 / +0.23; dL_r +0.69 / -0.90 / +0.41 / -0.61 / +0.85 / +0.22; fsel_r -0.60 / +0.83 /
  -0.47 / +0.55 / -0.64 / -0.27 (0.098); fsel_w +0.29 (0.076) / -0.15 / +0.33 (0.038) / -0.29 (0.071) / +0.18 /
  +0.48 (0.002); p < 0.001 where not given. Screens (D rho 30, gradient side): equal q_f per target on down_up_o at
  qf 0.045: Q_f 0.045, selectivity 1.100, mean selectivity 1.174, dL_f / dL_r 10.43 / 9.68, rank sum 22432 (D m 1.25:
  0.0407, 1.112, 1.201, 10.97 / 10.37, 31649); at qf 0.04 plain / C / D(0.045) selectivity 1.042 / 1.073 / 1.100.
  down_proj only: m 1.25 / 2 / 3 / 4: selectivity 1.265 / 1.245 / 1.218 / 1.193, dL_f 3.39 / 5.08 / 6.68 / 6.80, dL_r
  2.53 / 4.18 / 5.90 / 6.09, dL_ra 0.80 / 2.56 / 4.74 / 5.08, dL_wf -0.86 / 0.57 / 1.88 / 2.12, rank sum 13113 / 20982
  / 31473 / 41964; qf 0.03 / 0.06 / 0.1: dL_f 0.95 / 2.21 / 3.86, dL_r 0.64 / 1.61 / 3.01. down + up at m 1.25:
  selectivity 1.112, dL_f / dL_r 9.22 / 8.58.
  Finding: EM and FQ follow the edit magnitude like Memorization (FQ almost exactly: mean q_f +0.96), Utility the
  opposite way, so Overall is an inverted U in magnitude; the only proxy with a significant positive Overall
  correlation is the functional selectivity against pre-TOFU knowledge, dL_f / dL_world (+0.48). Equal q_f per
  target is less selective than the uniform rank multiple (it takes budget from down_proj). down_proj alone is the
  most selective edit seen (1.19-1.27) and the only one that leaves the model functional (dL_world ~0 at m 1.25; at
  m 3 the same rank sum as D m 1.25 on down_up_o, dL_f 6.7 instead of 11.0).
- [x] 01:11 Decision (from "carry out the proposals", 00:06): train the best candidates of the screen with phase two as
  D m 1.25 (restricted_semu free_all, lr 1e-5 constant, nll, seed 0), with TRAJECTORY=nll: D on down_proj only at m 2
  and m 3 (m 3 = the rank sum of D m 1.25 on down_up_o), and the forget-output budget qf 0.045 on down_up_o (the Q_f
  of D m 1.25 with equal q_f per target).
- [x] 00:16 Decision (user): track cheap training metrics over phase two (before it and after every epoch), including
  the loss, not the full Table 3 scores per epoch; the cheapest sets.
- [x] 00:16 Code: `recast_tofu.py --eval_trajectory nll` (launcher `TRAJECTORY=nll` in --export, run-name suffix
  `_traj`): before phase two and at every epoch end, the answer-token NLL and exact memorization (per-item
  teacher-forced token accuracy, as metric_em, in the training dtype) of forget, retain_perturbed (the 400 retain
  items of Model Utility), real authors and world facts, plus the epoch's mean training loss; random states
  restored so training is unchanged; control: points at epochs 0..N. `results.phase_two.log_history` keeps the
  Trainer log (loss, grad_norm, learning rate). The collector treats `eval_trajectory` as bookkeeping. Staged in the
  scratchpad and installed at 00:16, after every task of 25232879 / 25232883 had loaded the previous file.
  SHA-256 prefixes recast_tofu, collect_recast_tofu: c61f8de7 ce034f09 . Gate: `py_compile`, `bash -n`.
- [x] 00:39 Decision (user): module-type selectivity at shrink strengths other than s_min 0.3.
  Note: with alignment ~0.001 every row sits at the floor, so D = (s_min - 1) P(W), q_f and q_r both scale by
  (1 - s_min)^2 and the layer selectivity q_f / q_r does not depend on s_min (25232884: plain EVR s_min 0.3 / 0.2
  selectivity 1.0452 / 1.0453, Q_f ratio 1.304 = 0.64 / 0.49); only the functional (NLL) proxies can differ.
- [x] 00:39 Code: profiles `attn_q`, `attn_k`, `attn_v`, `attn_o` (one projection type each), test extended;
  proxy launcher `CONFIG_SET=types` (each type alone, plain and D m 1.25, s_min 0.3 / 0.5 / 0.7 / 0.9; gradients
  and covariances over decoder_linear; no trained join) and output name `warmstart_proxy_<set>_<job>.json`;
  `warmstart_proxy.py` stops before correlations with fewer than 3 joined configurations. Edited after 25232884 had
  loaded its code; no task pending. SHA-256 prefixes recast_core, warmstart_proxy, tests: 0daab925 02a2880b
  1d64d7e5 . Gate: `py_compile`, `bash -n`.
- [x] 00:39 Module type x shrink screen: job 25233350 (56 configurations, training-free).
  Result (01:23): COMPLETED in 43:46; NLL restore control passed (output `warmstart_proxy_types_25233350.json`). Layer
  selectivity is the same at every s_min, as derived (D: q 1.064-1.065, k 1.079, v 1.107, o 1.101, gate 1.081, up
  1.103, down 1.265; plain q 1.003, k 1.008, down 1.093), while Q_f scales with (1 - s_min)^2. Functional, D at s_min
  0.3 / 0.5 / 0.7 (dL_f, fsel_r = dL_f / dL_r): q_proj 7.65 1.003 / 4.50 1.03 / 1.04 1.20; k_proj 3.92 1.08 / 1.23
  1.14 / 0.20 1.21; v_proj 0.36 1.20 / 0.15 1.14 / 0.04 1.09; o_proj 0.82 1.17 / 0.29 1.13 / 0.07 1.05; gate 1.26
  1.10 / 0.80 1.11 / 0.32 1.13; up 5.63 1.14 / 2.00 1.19 / 0.35 1.22; down 3.39 1.34 / 1.44 1.50 / 0.31 1.45 (plain
  down 8.05 1.05 / 4.87 1.11 / 1.19 1.22). At s_min 0.9 every dL is <= 0.06 nat (ratios noise). v, o and gate
  lower the pre-TOFU NLL at every s_min (dL_ra, dL_wf < 0); q and k at s_min 0.3 alone raise every NLL by 4-9 nat.
  Finding: the functional selectivity depends on the shrink strength although the layer selectivity does not: a
  strong shrink saturates (the model breaks, forget and retain alike), a weak one shows the layer contrast. down_proj
  with basis D is the most selective at every strength, best at s_min 0.5 (1.50, dL_f 1.44, pre-TOFU NLL lower);
  q_proj and k_proj are the least selective and the most destructive.
- [x] 01:11 Training with trajectory (TRAJECTORY=nll), D rho 30 gradient side, seed 0, s_min 0.3, EVR 0.97 reference ranks:
  array 25233964 (mlp_down, m 2 and m 3) and array 25233965 (down_up_o, qf 0.045).
  Result (01:31): all three COMPLETED (14:23, 18:17, 19:21); 33 parity tests passed, frozen tensors moved 0, every
  free group changed, capture, representation (ratio 1.359 / 1.286 / 1.187 vs plain 1.111 / 1.106 / 1.089) and
  trajectory controls passed; qf budget control passed (q_f 0.0451 / 0.0450 / 0.0451 for o / up / down at median
  ranks 263 / 782 / 357 against EVR ranks 298 / 629 / 656). Mem / Util / Priv / EM / FQ / Overall:
  down_proj m 2: 0.9725 / 0.8803 / 0.8949 / 0.0397 / -0.7424 / 0.9142; down_proj m 3: 0.9553 / 0.8182 / 0.8762 /
  0.0196 / -0.4919 / 0.8797; down_up_o qf 0.045: 0.7547 / 0.9214 / 0.6561 / 0.1821 / -6.1685 / 0.7625 (D m 1.25 on
  down_up_o seeds 0 / 1 / 2: 0.9225 / 0.9110 / 0.9131). Trajectory (answer NLL / EM, fp32) for down_proj m 2: forget
  5.21 / 0.272 at epoch 0, 1.59 / 0.649 at epoch 1, then rising to 2.53 / 0.612 at epoch 10; retain 4.44 -> 0.82 ->
  0.017; real authors 7.30 -> 4.57 (below Init Fine. 4.74) -> 6.37; world facts 7.28 -> 5.45 -> 8.08; training loss
  1.76 (epoch 1) -> 0.029. m 3 and qf 0.045 follow the same shape (forget NLL minimum at epoch 1-2: 1.85 / 1.22, then
  2.88 / 2.07 at epoch 10). The trajectory EM agrees with the TOFU EM (down m 2: 0.612 fp32 vs 0.601 from the
  rescaled score).
  Finding: down_proj alone at m 2 matches the best configuration (0.914 vs 0.911-0.922) with higher Memorization and
  Utility and lower Privacy; a larger down_proj budget (m 3) loses Utility. The equal-q_f budget fails although its
  warm start is as destructive as D m 1.25 (forget NLL 10.6): it halves the down_proj ranks, and phase two restores
  the forget answers (EM 0.67, Mem 0.755). What survives phase two is the down_proj edit, not the overall damage.
  Over phase two the first epoch relearns most of what the warm start broke, the forget answers included; later
  epochs fit retain and push the forget and the pre-TOFU NLL back up.
- [x] 02:21 Decisions (user): seeds for D on down_proj only at m 2 (s_min 0.3), and down_proj only at s_min 0.5 with
  larger ranks; a shorter phase two / an earlier lr step only explained, not run. Noted (user): the EVR-then-m rank
  rule uses two hyperparameters for one budget.
- [x] 02:21 D rho 30 gradient side on mlp_down, TRAJECTORY=nll, phase two as before: array 25236082 (s_min 0.3, m 2, seeds 1
  and 2) and array 25236083 (s_min 0.5, m 2 / 3 / 4, seed 0).
  Result (02:48): all five COMPLETED (14:26-26:42); 33 parity tests passed, frozen tensors moved 0, capture,
  representation and trajectory controls passed. Mem / Util / Priv / EM / FQ / Overall: s_min 0.3 m 2 seed 1 0.9717 /
  0.9066 / 0.9015 / 0.0388 / -0.6125 / 0.9255, seed 2 0.9776 / 0.9115 / 0.9027 / 0.0313 / -0.8108 / 0.9294; s_min 0.5
  m 2 0.8705 / 0.9401 / 0.5440 / 0.1815 / -3.2341 / 0.7406, m 3 0.8968 / 0.9205 / 0.7096 / 0.1445 / -2.6394 / 0.8309,
  m 4 0.9047 / 0.9267 / 0.6884 / 0.1313 / -2.0999 / 0.8248. Collector (job 25236340, 00:00:05, output
  `runtime_artifacts/ai-engram/recast_tofu/summary_20261027_0248.json`): down_proj only m 2, n = 3: Mem 0.9739 +-
  0.0032, Util 0.8995 +- 0.0168, Priv 0.8997 +- 0.0042, EM 0.0366 +- 0.0046, FQ -0.7219 +- 0.1007, Overall 0.9230 +-
  0.0079 (D m 1.25 on down_up_o, n = 3: 0.9506 / 0.8647 / 0.9361 / 0.0144 / -0.8821 / 0.9155 +- 0.0061). Trajectory
  forget answer NLL / EM, epoch 0 -> 1 -> 10: s_min 0.5 m 2 2.34 / 0.563 -> 1.00 / 0.742 -> 1.88 / 0.667; m 3 2.93 ->
  1.16 -> 2.14; m 4 2.97 -> 1.19 -> 2.12; s_min 0.3 m 2 seeds 1 / 2 5.21 -> 1.59 -> 2.55 / 2.56 (seed 0 2.53).
  Finding: down_proj alone at m 2 is the best mean of the campaign (0.923, +0.008 over D m 1.25 on down_up_o, about
  one standard deviation): Utility +0.035 and Memorization +0.023 against Privacy -0.036. The weaker shrink (s_min
  0.5) loses at every rank although its warm start was the most selective one in the training-free screen: it
  breaks the forget answers less (forget NLL 2.3-3.0 instead of 5.2), phase two brings more of them back (EM 0.65-0.67
  instead of 0.61), and a larger rank does not make up for it (m 3 and m 4 ~0.83). The functional selectivity right
  after the warm start does not predict the trained result; the strength of the down_proj edit does.
- [x] 03:48 Decision (user): carry out the three proposed steps: (1) down_proj only at m 2 with a stronger shrink (s_min
  0.2, 0.1); (2) a shorter phase two (3 and 5 epochs) and a step of the lr to 1e-6 after epoch 2; (3) one budget
  parameter: m 1 with the EVR whose ranks match m 2 (down_proj only) and m 1.25 (down_up_o), first located without
  training.
- [x] 03:48 Code: proxy launcher `CONFIG_SET=evr` (D m 2 on mlp_down, D m 1.25 on down_up_o, and D m 1 at EVR 0.98 /
  0.985 / 0.99 / 0.993 / 0.995 / 0.998 on both). Gate: `bash -n`. No Python change.
- [x] 03:49 Jobs, D rho 30 gradient side on mlp_down at m 2, seed 0, TRAJECTORY=nll, phase two as before unless stated:
  25243024 (proxy, CONFIG_SET=evr); 25243028 (s_min 0.2 and 0.1); 25243032 (s_min 0.3, EPOCHS=3); 25243033 (s_min 0.3,
  EPOCHS=5); 25243034 (s_min 0.3, 10 epochs, lr points 0:0,1:1e-5,2:1e-5,2:1e-6,10:1e-6). The 3- and 5-epoch runs keep
  the lr name warm1_const1e-5_e10 in their run names (the schedule is the same constant 1e-5; only --epochs differs).
  Result (04:20): all COMPLETED (25243028 13:56 / 16:25, 25243032 9:09, 25243033 16:15, 25243034 24:22); in each, 33
  parity tests passed, frozen tensors moved 0, capture and trajectory controls passed. Mem / Util / Priv / EM / FQ /
  Overall: s_min 0.2 0.9861 / 0.8244 / 0.9078 / 0.0189 / -0.0261 / 0.9013; s_min 0.1 0.9638 / 0.8291 / 0.8308 /
  0.0680 / -0.1935 / 0.8702; 3 epochs 0.9431 / 0.9287 / 0.4286 / 0.0810 / -1.1856 / 0.6711; 5 epochs 0.9637 / 0.9074 /
  0.7247 / 0.0496 / -0.5510 / 0.8524; lr step to 1e-6 after epoch 2 0.8963 / 0.8970 / 0.8089 / 0.0501 / -1.4369 /
  0.8654 (10 epochs at constant 1e-5, seed 0: 0.9725 / 0.8803 / 0.8949 / 0.0397 / -0.7424 / 0.9142; n = 3 0.9230).
  Control: the 3- and 5-epoch trajectories equal the 10-epoch run of 25233964_0 at epochs 0-3 / 0-4 to four decimals
  (forget NLL 5.2142, 1.5918, 1.6796, 1.9306, 2.1468), so these runs are its epoch-3 and epoch-5 states. Forget answer
  NLL / EM at epochs 0, 1, 10: s_min 0.2 6.84 / 0.135, 1.83 / 0.615, 2.82 / 0.590; s_min 0.1 8.18 / 0.049, 2.05 /
  0.585, 3.04 / 0.572; lr step 5.21, 1.59, 2.81 / 0.615.
  Finding: for down_proj alone the shrink optimum is near s_min 0.3 (0.5 / 0.3 / 0.2 / 0.1: 0.74 / 0.92 / 0.90 /
  0.87): stronger shrinks forget more (FQ -0.03 at 0.2) but cost Utility (0.82-0.83). Privacy needs the full phase
  two (0.43 / 0.72 / 0.89 after 3 / 5 / 10 epochs) while Utility falls only slightly (0.93 / 0.91 / 0.88), so a
  shorter phase two or a smaller late lr loses; the early-epoch gain in pre-TOFU NLL does not carry the scores.
- [x] 04:10 Proxy CONFIG_SET=evr: job 25243024. Result (04:10): COMPLETED in 19:50; NLL restore control passed (output
  `warmstart_proxy_evr_25243024.json`). Rank sums at m 1, EVR 0.98 / 0.985 / 0.99 / 0.993 / 0.995 / 0.998: down_proj
  only 12633 / 14153 / 16253 / 18027 / 19615 / 23434 (m 2 at EVR 0.97: 20982); down_up_o 30968 / 35054 / 40826 / 45827
  / 50405 / 61845 (m 1.25: 31649). Q_f, selectivity, dL_f / dL_r: down_proj m 2 0.1798, 1.245, 5.08 / 4.18 against
  EVR 0.995 0.1669, 1.249, 5.33 / 4.35 and 0.998 0.2109, 1.240, 6.44 / 5.49; down_up_o m 1.25 0.0407, 1.112, 10.97 /
  10.37 against EVR 0.98 0.0395, 1.114, 11.03 / 10.48. Chosen single-parameter budgets: EVR 0.996 for down_proj only
  (rank sum interpolated ~20.9k) and EVR 0.98 for down_up_o, both at m 1.
- [x] 04:10 One budget parameter (m 1, D rho 30 gradient side, s_min 0.3, seed 0, TRAJECTORY=nll, phase two as before):
  25243379 (mlp_down, EVR 0.996) and 25243380 (down_up_o, EVR 0.98).
  Result (04:37): both COMPLETED (14:53, 24:46); 33 parity tests passed, frozen tensors moved 0, capture and
  trajectory controls passed. down_proj only, EVR 0.996 (median down_proj rank 1491; m 2 at EVR 0.97: 1564): 0.9633
  / 0.9026 / 0.8832 / 0.0530 / -0.3297 / 0.9151 (m 2: seed 0 0.9142, n = 3 0.9230 +- 0.0079); down_up_o, EVR 0.98
  (median down_proj rank 942): 0.9459 / 0.8765 / 0.9215 / 0.0204 / -1.5250 / 0.9137 (m 1.25: seed 0 0.9225, n = 3
  0.9155 +- 0.0061).
  Finding: one EVR threshold at m 1 reproduces the m-scaled budgets within seed noise, so the rank multiplier is not
  needed as a second hyperparameter; the EVR alone sets the budget (0.996 for down_proj only, 0.98 for down_up_o).
