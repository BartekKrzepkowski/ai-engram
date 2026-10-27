#!/bin/bash
# ReCAST on TOFU forget10, scored like Table 3: one array task per combination of the axes below,
# decoded as a mixed-radix number with the ablation innermost and the variant outermost; EVR applies
# to the restricted variants only. The lm_head is not shrunk; head and input
# embedding stay one tied matrix, trained in phase two. Each task first runs the CPU parity
# tests of recast_core.py and stops if they fail.
# Submit from the repository root, stating the two phase-two choices explicitly:
#   sbatch --export=ALL,EPOCH_LENGTH=retain,TRAIN_DTYPE=float32,MAX_GRAD_NORM=1.0 replication/scripts/bash/recast_tofu/recast_tofu_gh200.sh
# MAX_GRAD_NORM=0 disables gradient clipping in the Hugging Face Trainer. EPOCHS=0 in --export evaluates the
# warm start alone (no phase two); the default is 10. TRAJECTORY=nll in --export also tracks the answer NLL and
# exact memorization of forget, retain_perturbed, real authors and world facts before phase two and after every epoch.
# COMPONENT_DIAGNOSTICS=yes in --export writes the per-target update components and the first-order loss check.
# BASE_ID=<model id or directory> (with BASE_TAG=<name for the run name>) in --export starts from another model than
# Init Fine., e.g. a saved Engram edit; with VARIANTS=(unrestricted) and S_MINS=(1.0) phase one is the identity and the
# run is phase two alone. SAVE_MODEL_SEED=<seed> in --export saves that seed's model after phase two.
#SBATCH --job-name=ai-engram-recast
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0
#SBATCH --output=replication/logs/slurm/%x-%A_%a.out

set -euo pipefail

: "${EPOCH_LENGTH:?set EPOCH_LENGTH=forget|retain with sbatch --export}"
: "${TRAIN_DTYPE:?set TRAIN_DTYPE=bfloat16|float32 with sbatch --export}"
: "${MAX_GRAD_NORM:?set MAX_GRAD_NORM (0 = no clipping) with sbatch --export}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/replication/scripts/recast_tofu.py" ]] || { echo "ERROR: submit from the repository root." >&2; exit 1; }
STORAGE=/net/storage/pr3/plgrid/plggdnnp
ENV_PREFIX="${AI_ENGRAM_ENV_PREFIX:-${STORAGE}/conda_envs/ai-engram-gh200}"
PY="${ENV_PREFIX}/bin/python"

export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${STORAGE}/cache/${USER}/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=ai-engram-replication
export WANDB_DIR="${STORAGE}/archive/${USER}/ai-engram_logs"
export TOKENIZERS_PARALLELISM=false

# Current configuration; the decisions behind each value and every earlier setting are in
# replication/docs/WORK_LOG.md. Axes, outermost first: variant, profile, floor, EVR, seed,
# learning rate, retain loss, retain input filter, ablation.
# restricted_shrink: the restricted warm start without adapters, phase two on every parameter.
VARIANTS=(restricted_shrink)
PROFILES=(down_up_o)
S_MINS=(0.3)
EVRS=(0.97)
SEEDS=(0)
# Learning-rate schedules, peak|points|name: warmup to the peak in epoch 1, the peak until epoch 5,
# then a step down to 1e-5 (user, 2026-10-22). Runs up to 24990942 went linearly from the peak at
# epoch 1 to the epoch-5 value instead.
LR_CONFIGS=(
    "1e-5|0:0,1:1e-5,10:1e-5|warm1_const1e-5_e10"
)
# Phase-two retain loss, loss/KL weight: nll, or kl / nll_kl with the weight of KL to Init Fine.
RETAIN_LOSSES=(nll)
# Retain input covariance treatment (restricted only), kind/parameter: none, ridge/<kappa> (filter A of
# the warm-start delta), whiten/<rho> (restriction bases from the retain-whitened forget gradient, C) or
# repr/<rho>/<output side>[/<beta>] (bases D from the forget-vs-retain input covariance contrast; output side
# identity, gradient, whitened_gradient or representation; beta mixes the forget activations (1, default)
# with the forget gradient (0) on the input side; needs a RANK_MULTS budget).
# (24991502 ran ridge/0.1, ridge/1, ridge/10; 24993357 ridge/30-1000 and whiten/0.01-10.)
FILTERS=(repr/30/gradient)
# Edit budget (restricted only): none (rank from EVR), m (rank = round(m * plain-basis rank at EVR) per
# target, the same for the plain and the whitened basis) or qf/<q> (per target the smallest rank whose
# warm-start delta changes forget outputs by tr(D C_f D^T) / tr(W C_f W^T) >= q).
RANK_MULTS=(1.25)
# Independent ablations against the reference (full_space alignment, forget_only source, no
# layer scaling): each changes exactly one factor. Format: scope:source:scaling:phase2_free:tag.
# phase2_free (restricted_semu only, ignored otherwise): none, norms, attention, mlp or all (every
# parameter without an adapter; the default since 2026-10-22). (24979138 ran align_subspace,
# source_full_data, scaling_subspace_norm; 24982024 scaling_subspace_norm_mean; 24987066 reference,
# scaling_subspace_norm_mean, free_norms, free_attention, free_all with phase2_free none for the first two;
# 24987255, 24989313, 24989314 restricted_shrink with tag reference.)
ABLATIONS=(
    "full_space:forget_only:none:all:free_all"
)
# Components of the projected update (restricted only, innermost axis): none (the original update), full (its SVD
# reconstruction), signed (only components with c_j = <G_f, D_j> > 0), random (as many, drawn with the seed) or
# uniform_matched (the original update scaled to the signed update's norm).
COMPONENT_MODES=(signed)
# Warm-start variants (restricted only, next-to-innermost axis): none, scale/<t> (W_Init + t (W* - W_Init)),
# scale/holdout (t chosen by forward passes so that the forget loss equals the Init Fine. holdout loss),
# types/<projection>=<t>,... (scale/<t> per projection type, every type of the profile listed) or
# dirw/contrast (per-direction shrink weighted by the forget / retain input-energy contrast).
WARM_STARTS=(scale/1.428571)
BLOCKS=all
EPOCHS="${EPOCHS:-10}"
TASK="${SLURM_ARRAY_TASK_ID:?run as an array task}"
i=${TASK}
COMP="${COMPONENT_MODES[$((i % ${#COMPONENT_MODES[@]}))]}"; i=$((i / ${#COMPONENT_MODES[@]}))
IFS=/ read -r WS_KIND WS_VALUE <<< "${WARM_STARTS[$((i % ${#WARM_STARTS[@]}))]}"; i=$((i / ${#WARM_STARTS[@]}))
IFS=: read -r SCOPE SOURCE SCALING FREE ABL_TAG <<< "${ABLATIONS[$((i % ${#ABLATIONS[@]}))]}"; i=$((i / ${#ABLATIONS[@]}))
IFS=/ read -r FILTER KAPPA FOUT FMIX <<< "${FILTERS[$((i % ${#FILTERS[@]}))]}"; i=$((i / ${#FILTERS[@]}))
RANK_MULT="${RANK_MULTS[$((i % ${#RANK_MULTS[@]}))]}"; i=$((i / ${#RANK_MULTS[@]}))
IFS=/ read -r RETAIN_LOSS KL_WEIGHT <<< "${RETAIN_LOSSES[$((i % ${#RETAIN_LOSSES[@]}))]}"; i=$((i / ${#RETAIN_LOSSES[@]}))
IFS='|' read -r LR_PEAK LR_POINTS LR_NAME <<< "${LR_CONFIGS[$((i % ${#LR_CONFIGS[@]}))]}"; i=$((i / ${#LR_CONFIGS[@]}))
SEED="${SEEDS[$((i % ${#SEEDS[@]}))]}"; i=$((i / ${#SEEDS[@]}))
EVR="${EVRS[$((i % ${#EVRS[@]}))]}"; i=$((i / ${#EVRS[@]}))
S_MIN="${S_MINS[$((i % ${#S_MINS[@]}))]}"; i=$((i / ${#S_MINS[@]}))
PROFILE="${PROFILES[$((i % ${#PROFILES[@]}))]}"; i=$((i / ${#PROFILES[@]}))
VARIANT="${VARIANTS[${i}]:?task ${TASK} is beyond the configured axes}"
LOSS_ARGS=(--retain_loss "${RETAIN_LOSS}")
LOSS_TAG="${RETAIN_LOSS}"
if [[ "${RETAIN_LOSS}" == *kl* ]]; then
    LOSS_ARGS+=(--kl_weight "${KL_WEIGHT:?give the KL weight as loss/weight}")
    LOSS_TAG="${RETAIN_LOSS}${KL_WEIGHT/./p}"
fi
FILTER_TAG="filter_${FILTER}${KAPPA:+${KAPPA/./p}}${FOUT:+_${FOUT}}${FMIX:+_mix${FMIX/./p}}"
case "${RANK_MULT}" in
    none) ;;
    qf/*) QF="${RANK_MULT#qf/}"; FILTER_TAG="${FILTER_TAG}_qf${QF/./p}" ;;
    *) FILTER_TAG="${FILTER_TAG}_rankx${RANK_MULT/./p}" ;;
esac
EP_TAG=""
[[ "${EPOCHS}" == "0" ]] && EP_TAG="_warmstartonly"
[[ "${COMP}" != none ]] && FILTER_TAG="${FILTER_TAG}_comp${COMP}"
if [[ "${WS_KIND}" == types ]]; then
    FILTER_TAG="${FILTER_TAG}_types_$(sed 's/_proj//g; s/\./p/g; s/=//g; s/,/_/g' <<< "${WS_VALUE}")"
elif [[ "${WS_KIND}" != none ]]; then
    FILTER_TAG="${FILTER_TAG}_${WS_KIND}${WS_VALUE/./p}"
fi
BASE_ID="${BASE_ID:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}"
BASE_TAG="${BASE_TAG:-}"
[[ "${BASE_ID}" != open-unlearning/tofu_Llama-3.2-1B-Instruct_full && -z "${BASE_TAG}" ]] && { echo "ERROR: set BASE_TAG for BASE_ID=${BASE_ID}" >&2; exit 1; }
TRAJ_ARGS=()
if [[ -n "${TRAJECTORY:-}" ]]; then
    TRAJ_ARGS=(--eval_trajectory "${TRAJECTORY}")
    EP_TAG="${EP_TAG}_traj"
fi
S_TAG="${S_MIN/./p}"
E_TAG="${EVR/./p}"
CLIP_TAG="${MAX_GRAD_NORM/./p}"
[[ "${MAX_GRAD_NORM}" == "0" ]] && CLIP_TAG=none
# Init Fine. and Retain raw metrics: Table 3 run 24952059 (same evaluation code and environment).
ANCHORS_RAW="${STORAGE}/experiment_results/ai-engram/table3/table3_forget10_seed0_24952059/raw_metrics.json"
RUN_NAME="recast_${BASE_TAG:+${BASE_TAG}_}${VARIANT}_${PROFILE}_blocks${BLOCKS}_tied_smin${S_TAG}_evr${E_TAG}_${ABL_TAG}_${FILTER_TAG}_lr${LR_NAME}_${LOSS_TAG}_seed${SEED}_${EPOCH_LENGTH}_${TRAIN_DTYPE}_clip${CLIP_TAG}${EP_TAG}_${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
OUT_DIR="${STORAGE}/experiment_results/ai-engram/recast_tofu/${RUN_NAME}"
SAVE_ARGS=()
if [[ -n "${SAVE_MODEL_SEED:-}" && "${SEED}" == "${SAVE_MODEL_SEED}" ]]; then
    SAVE_ARGS=(--save_model_dir "${STORAGE}/runtime_artifacts/ai-engram/recast_tofu/models/${RUN_NAME}")
fi

VARIANT_ARGS=(--freeze_input_embedding no --shrink_layer_scaling "${SCALING}")
VARIANT_TAGS=(variant_unrestricted "scaling_${SCALING}")
if [[ "${VARIANT}" == restricted_* ]]; then
    VARIANT_ARGS=(--restriction_source "${SOURCE}" --restriction_evr "${EVR}"
                  --restriction_alignment_scope "${SCOPE}" --shrink_layer_scaling "${SCALING}")
    case "${FILTER}" in
        none) VARIANT_ARGS+=(--retain_input_filter none --restriction_whitening none --restriction_representation none) ;;
        ridge) VARIANT_ARGS+=(--retain_input_filter ridge --retain_filter_kappa "${KAPPA:?give kappa as ridge/kappa}"
                              --restriction_whitening none --restriction_representation none) ;;
        whiten) VARIANT_ARGS+=(--retain_input_filter none --restriction_whitening retain_input
                               --whitening_rho "${KAPPA:?give rho as whiten/rho}" --restriction_representation none) ;;
        repr) VARIANT_ARGS+=(--retain_input_filter none --restriction_whitening none
                             --restriction_representation "${FOUT:?give the output side as repr/rho/side}"
                             --whitening_rho "${KAPPA:?give rho as repr/rho/side}" ${FMIX:+--representation_mix "${FMIX}"}) ;;
        *) echo "ERROR: unknown retain input treatment ${FILTER}" >&2; exit 1 ;;
    esac
    case "${RANK_MULT}" in
        none) ;;
        qf/*) VARIANT_ARGS+=(--restriction_forget_output_target "${RANK_MULT#qf/}") ;;
        *) VARIANT_ARGS+=(--restriction_rank_multiplier "${RANK_MULT}") ;;
    esac
    [[ "${COMP}" != none ]] && VARIANT_ARGS+=(--component_selection "${COMP}")
    case "${WS_KIND}" in
        none) ;;
        scale) if [[ "${WS_VALUE}" == holdout ]]; then VARIANT_ARGS+=(--warm_start_scale_target holdout)
               else VARIANT_ARGS+=(--warm_start_scale "${WS_VALUE:?give t as scale/t}"); fi ;;
        types) VARIANT_ARGS+=(--warm_start_scale_by_type "${WS_VALUE:?give types/<projection>=<t>,...}") ;;
        dirw) VARIANT_ARGS+=(--direction_weighting "${WS_VALUE:?give dirw/contrast}") ;;
        *) echo "ERROR: unknown warm-start variant ${WS_KIND}" >&2; exit 1 ;;
    esac
    [[ "${COMPONENT_DIAGNOSTICS:-}" == yes ]] && VARIANT_ARGS+=(--component_diagnostics yes)
    VARIANT_TAGS=("variant_${VARIANT}" "source_${SOURCE}" "evr_${E_TAG}" "align_${SCOPE}" "scaling_${SCALING}" "${ABL_TAG}"
                  "${FILTER_TAG}")
    if [[ "${VARIANT}" == "restricted_semu" ]]; then
        VARIANT_ARGS+=(--phase2_free "${FREE}")
        VARIANT_TAGS+=("phase2_free_${FREE}")
    fi
fi

echo "[launch] host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-none}_${TASK} git=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown) branch=$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo unknown)"
echo "[launch] base=${BASE_ID} variant=${VARIANT} profile=${PROFILE} s_min=${S_MIN} epoch_length=${EPOCH_LENGTH} train_dtype=${TRAIN_DTYPE} max_grad_norm=${MAX_GRAD_NORM} out_dir=${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[launch] nvidia-smi unavailable"

cd "${REPO_ROOT}"
echo "[gate] recast_core parity tests (CPU)"
"${PY}" -m pytest -q -p no:cacheprovider replication/tests/test_recast_core.py

"${PY}" -u replication/scripts/recast_tofu.py \
    --base_id "${BASE_ID}" \
    --dataset_id locuslab/TOFU \
    --forget_split forget10 \
    --retain_split retain90 \
    --holdout_split holdout10 \
    --train_dtype "${TRAIN_DTYPE}" \
    --eval_dtype bfloat16 \
    --attn_implementation sdpa \
    --variant "${VARIANT}" \
    --targets "${PROFILE}" \
    --target_blocks "${BLOCKS}" \
    --untie_lm_head no \
    --train_lm_head yes \
    --s_min "${S_MIN}" \
    --grad_batch_size 8 \
    "${VARIANT_ARGS[@]}" \
    --epochs "${EPOCHS}" \
    "${TRAJ_ARGS[@]}" \
    --epoch_length "${EPOCH_LENGTH}" \
    --per_device_batch_size 8 \
    --grad_accum 4 \
    --lr "${LR_PEAK}" \
    --weight_decay 0.01 \
    --lr_scheduler piecewise_epochs \
    --lr_points "${LR_POINTS}" \
    --optim adamw_torch \
    "${LOSS_ARGS[@]}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --seed "${SEED}" \
    --eval_batch_size 16 \
    --anchors_raw "${ANCHORS_RAW}" \
    --out_dir "${OUT_DIR}" \
    "${SAVE_ARGS[@]}" \
    --run_name "${RUN_NAME}" \
    --wandb_mode online \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_tags recast_tofu_comparison matched_protocol method_recast \
        model_tofu_llama3p2_1b_instruct split_forget10 "base_${BASE_TAG:-init_fine}" "${VARIANT_TAGS[@]}" "smin_${S_TAG}" \
        phase2_retain_only "retain_loss_${LOSS_TAG}" optim_adamw_torch "lr_peak_${LR_PEAK}" "lr_${LR_NAME}" wd_0p01 "epochs_${EPOCHS}" "epoch_len_${EPOCH_LENGTH}" \
        "dtype_${TRAIN_DTYPE}" grad_views_1 "targets_${PROFILE}" "blocks_${BLOCKS}" head_no_shrink_trained lm_head_tied embedding_trained "grad_clip_${CLIP_TAG}" "seed_${SEED}"

echo "[launch] done"
