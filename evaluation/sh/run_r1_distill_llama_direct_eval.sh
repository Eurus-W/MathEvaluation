#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
    cat <<'EOF'
Usage:
  bash sh/run_r1_distill_llama_direct_eval.sh <setting_type> <model_name_or_path> [cuda_devices]

Example:
  bash sh/run_r1_distill_llama_direct_eval.sh r1_distill_llama deepseek-ai/DeepSeek-R1-Distill-Llama-8B 5
  bash sh/run_r1_distill_llama_direct_eval.sh r1_distill_qwen /path/to/DeepSeek-R1-Distill-Qwen-7B "0,1"
  bash sh/run_r1_distill_llama_direct_eval.sh qwen3_4b /path/to/Qwen3-4B-Thinking-2507 6

Defaults:
  - setting_type controls default prompt/top_k/max_tokens:
      r1_distill_llama -> prompt=llama-r1-distill-cot, top_k=-1, max_tokens=16384
      r1_distill_qwen  -> prompt=qwen25-math-cot,      top_k=-1, max_tokens=16384
      qwen3_4b         -> prompt=qwen25-math-cot,      top_k=20, max_tokens=81920
  - cot_baseline=none
  - truncation_ratio=1.0
  - temperature=0.6, n_sampling=1, seeds=0,1,2,3,4

Optional env overrides:
  DATA_NAMES, SEEDS, NUM_TEST_SAMPLE
  TEMPERATURE, N_SAMPLING, TOP_P
  PROMPT_TYPE_OVERRIDE, TOP_K_OVERRIDE, MAX_TOKENS_OVERRIDE
  PIPELINE_PARALLEL_SIZE, OVERWRITE, DRY_RUN
EOF
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        exit 0
    fi
    exit 1
fi

SETTING_TYPE=$1
MODEL_NAME_OR_PATH=$2
CUDA_DEVICES=${3:-0}

sanitize_name() {
    local value=$1
    value=${value%/}
    value=$(echo "${value}" | sed 's#/*$##' | awk -F/ '{if (NF >= 2) print $(NF-1) "/" $NF; else print $NF}')
    value=${value// /_}
    value=${value//\//_}
    echo "${value}"
}

resolve_setting() {
    local setting_type=$1
    case "${setting_type}" in
        r1_distill_llama)
            RESOLVED_PROMPT_TYPE="llama-r1-distill-cot"
            RESOLVED_TOP_K=-1
            RESOLVED_MAX_TOKENS=16384
            ;;
        r1_distill_qwen)
            RESOLVED_PROMPT_TYPE="qwen25-math-cot"
            RESOLVED_TOP_K=-1
            RESOLVED_MAX_TOKENS=16384
            ;;
        qwen3_4b)
            RESOLVED_PROMPT_TYPE="qwen25-math-cot"
            RESOLVED_TOP_K=20
            RESOLVED_MAX_TOKENS=81920
            ;;
        *)
            echo "[error] unknown setting_type=${setting_type}" >&2
            echo "[error] supported: r1_distill_llama, r1_distill_qwen, qwen3_4b" >&2
            exit 1
            ;;
    esac
}

resolve_setting "${SETTING_TYPE}"

PROMPT_TYPE=${PROMPT_TYPE_OVERRIDE:-${RESOLVED_PROMPT_TYPE}}
DATA_NAMES=${DATA_NAMES:-gsm8k,math_500,minerva_math,aime24,aime25,aime26,amc23}
SEEDS=${SEEDS:-0,1,2,3,4}
NUM_TEST_SAMPLE=${NUM_TEST_SAMPLE:--1}
TEMPERATURE=${TEMPERATURE:-0.6}
N_SAMPLING=${N_SAMPLING:-1}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K_OVERRIDE:-${RESOLVED_TOP_K}}
MAX_TOKENS=${MAX_TOKENS_OVERRIDE:-${RESOLVED_MAX_TOKENS}}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
OVERWRITE=${OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}

SETTING_TAG="t${TEMPERATURE}_p${TOP_P}_n${N_SAMPLING}"
if [[ ${TOP_K} -gt 0 ]]; then
    SETTING_TAG="${SETTING_TAG}_k${TOP_K}"
fi

MODEL_TAG=$(sanitize_name "${MODEL_NAME_OR_PATH}")
RUN_ROOT="${EVAL_DIR}/outputs/direct_eval_none_tr1"
LOG_ROOT="${EVAL_DIR}/logs/direct_eval_none_tr1"
OUTPUT_DIR="${RUN_ROOT}/settings/${SETTING_TYPE}/models/${MODEL_TAG}/${SETTING_TAG}"
LOG_DIR="${LOG_ROOT}/settings/${SETTING_TYPE}/models/${MODEL_TAG}/${SETTING_TAG}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[direct-eval] setting_type=${SETTING_TYPE}"
echo "[direct-eval] model_name_or_path=${MODEL_NAME_OR_PATH}"
echo "[direct-eval] prompt_type=${PROMPT_TYPE}"
echo "[direct-eval] data_names=${DATA_NAMES}"
echo "[direct-eval] seeds=${SEEDS}"
echo "[direct-eval] setting=${SETTING_TAG}"
echo "[direct-eval] top_k=${TOP_K}"
echo "[direct-eval] max_tokens=${MAX_TOKENS}"
echo "[direct-eval] cuda_devices=${CUDA_DEVICES}"
echo "[direct-eval] output_dir=${OUTPUT_DIR}"
echo "[direct-eval] log_dir=${LOG_DIR}"

print_dry_run_command() {
    local cuda_devices=$1
    shift
    printf '[dry_run] cmd=TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=%q ' "${cuda_devices}"
    printf '%q ' "$@"
    printf '\n'
}

IFS=',' read -r -a seed_items <<< "${SEEDS}"
for seed in "${seed_items[@]}"; do
    log_file="${LOG_DIR}/seed${seed}.log"
    cmd=(
        python3 -u math_eval.py
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --data_names "${DATA_NAMES}" \
        --output_dir "${OUTPUT_DIR}" \
        --split test \
        --prompt_type "${PROMPT_TYPE}" \
        --apply_chat_template \
        --cot_baseline none \
        --truncation_ratio 1.0 \
        --num_test_sample "${NUM_TEST_SAMPLE}" \
        --seed "${seed}" \
        --temperature "${TEMPERATURE}" \
        --n_sampling "${N_SAMPLING}" \
        --top_p "${TOP_P}" \
        --top_k "${TOP_K}" \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --pipeline_parallel_size "${PIPELINE_PARALLEL_SIZE}" \
        --max_tokens_per_call "${MAX_TOKENS}"
    )

    if [[ "${OVERWRITE}" == "1" ]]; then
        cmd+=(--overwrite)
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry_run] seed=${seed} log=${log_file}"
        print_dry_run_command "${CUDA_DEVICES}" "${cmd[@]}"
    else
        (
            cd "${EVAL_DIR}"
            TOKENIZERS_PARALLELISM=false \
            CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
            "${cmd[@]}"
        ) >"${log_file}" 2>&1
    fi
done

if [[ "${DRY_RUN}" != "1" && ${#seed_items[@]} -gt 1 ]]; then
    (
        cd "${EVAL_DIR}"
        python3 summarize_repeat_metrics.py \
            --output_dir "${OUTPUT_DIR}" \
            --data_names "${DATA_NAMES}" \
            --prompt_type "${PROMPT_TYPE}" \
            --temperature "${TEMPERATURE}" \
            --n_sampling "${N_SAMPLING}" \
            --seeds "${SEEDS}" \
            --model_name_or_path "${MODEL_NAME_OR_PATH}"
    )
fi

echo "[done] direct eval finished"
