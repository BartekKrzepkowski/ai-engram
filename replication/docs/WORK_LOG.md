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
