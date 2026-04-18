#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ $# -lt 2 ]]; then
    cat <<'EOF'
Usage:
  bash sh/run_dual_model_naive_baselines.sh <model_1> <model_2> [cuda_devices_model_1] [cuda_devices_model_2]

Notes:
  - Always passes --apply_chat_template; the tokenizer's own chat_template handles formatting.
  - Sweeps the TokenSkip naive baselines (BeConcise, OnlyNumbers, AbbreWords, LC-Prompt)
    via --cot_baseline, plus the Truncation baseline via --truncation_ratio.
  - Default benchmarks: math,minerva_math,olympiadbench,aime24.
  - Default decoding settings:
      1) temperature=0.0, n_sampling=1
      2) temperature=0.6, n_sampling=1
      3) temperature=1.0, n_sampling=16
  - Model 1 defaults to prompt_type=qwen25-math-cot; model 2 defaults to llama-r1-distill-cot.
    Override via PROMPT_TYPE_1 / PROMPT_TYPE_2 (prompt_type only affects stop-word logic here;
    the actual prompt is built from the tokenizer's chat_template).
  - If the two CUDA device arguments are different, the two models run in parallel.
  - If the two CUDA device arguments are the same, all jobs run sequentially on that single GPU.
  - Optional env overrides:
      DATA_NAMES="math,minerva_math,olympiadbench,aime24,amc23"
      SETTING_SPECS="0.0:1,0.6:1"
      SEEDS="0,1,2,3,4"
      SETTING_ALIAS="all"
      RUN_ORDER="benchmark_major"
      OVERWRITE=0
      BASELINES="none,beconcise,onlynumbers,abbrewords,lc_prompt"
      TRUNCATION_RATIOS="1.0"          # e.g. "1.0,0.9,0.7,0.5" to sweep Truncation
      LC_RATIO="0.5"
      PROMPT_TYPE_1="qwen25-math-cot"
      PROMPT_TYPE_2="llama-r1-distill-cot"
EOF
    exit 1
fi

MODEL_1=$1
MODEL_2=$2
CUDA_DEVICES_1=${3:-0}
CUDA_DEVICES_2=${4:-1}

PROMPT_TYPE_1=${PROMPT_TYPE_1:-"qwen25-math-cot"}
PROMPT_TYPE_2=${PROMPT_TYPE_2:-"llama-r1-distill-cot"}
DATA_NAMES=${DATA_NAMES:-"math,minerva_math,olympiadbench,aime24"}
SETTING_SPECS=${SETTING_SPECS:-"0.0:1,0.6:1,1.0:16"}
SEEDS=${SEEDS:-"0"}
SETTING_ALIAS=${SETTING_ALIAS:-""}
RUN_ORDER=${RUN_ORDER:-"seed_major"}
SPLIT="test"
NUM_TEST_SAMPLE=${NUM_TEST_SAMPLE:--1}
MAX_TOKENS=${MAX_TOKENS:-2048}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
OVERWRITE=${OVERWRITE:-1}
BASELINES=${BASELINES:-"none,beconcise,onlynumbers,abbrewords,lc_prompt"}
TRUNCATION_RATIOS=${TRUNCATION_RATIOS:-"1.0"}
LC_RATIO=${LC_RATIO:-"0.5"}

RUN_ROOT="${EVAL_DIR}/outputs/dual_model_naive_baselines"
LOG_ROOT="${EVAL_DIR}/logs/dual_model_naive_baselines"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

sanitize_name() {
    local value=$1
    value=${value%/}
    value=$(echo "${value}" | sed 's#/*$##' | awk -F/ '{if (NF >= 2) print $(NF-1) "/" $NF; else print $NF}')
    value=${value// /_}
    value=${value//\//_}
    echo "${value}"
}

resolve_setting_alias() {
    local alias_name=$1
    case "${alias_name}" in
        basic)
            echo "0.0:1:0;0.6:1:0;1.0:16:0"
            ;;
        t06x5)
            echo "0.6:1:0,1,2,3,4"
            ;;
        all)
            echo "0.0:1:0;0.6:1:0,1,2,3,4;1.0:16:0"
            ;;
        *)
            echo "[error] unknown SETTING_ALIAS=${alias_name}" >&2
            echo "[error] supported aliases: basic, t06x5, all" >&2
            exit 1
            ;;
    esac
}

run_repeat_summary() {
    local output_dir=$1
    local data_names_csv=$2
    local temperature=$3
    local n_sampling=$4
    local model_path=$5
    local seeds_csv=$6
    local prompt_type=$7
    local seed_count

    IFS=',' read -r -a seed_items <<< "${seeds_csv}"
    seed_count=${#seed_items[@]}

    if [[ ${seed_count} -le 1 ]]; then
        return 0
    fi

    (
        cd "${EVAL_DIR}"
        python3 summarize_repeat_metrics.py \
            --output_dir "${output_dir}" \
            --data_names "${data_names_csv}" \
            --prompt_type "${prompt_type}" \
            --temperature "${temperature}" \
            --n_sampling "${n_sampling}" \
            --seeds "${seeds_csv}" \
            --model_name_or_path "${model_path}"
    )
}

run_single_setting() {
    local model_path=$1
    local cuda_devices=$2
    local prompt_type=$3
    local baseline=$4
    local truncation_ratio=$5
    local temperature=$6
    local n_sampling=$7
    local seeds_csv=$8
    local seed
    local log_file
    local output_dir
    local log_suffix
    local -a seed_items
    local data_name

    local model_tag
    model_tag=$(sanitize_name "${model_path}")

    local setting_tag="t${temperature}_n${n_sampling}"
    local tr_tag="tr${truncation_ratio}"
    output_dir="${RUN_ROOT}/${model_tag}/${baseline}/${tr_tag}/${setting_tag}"

    mkdir -p "${output_dir}" "${LOG_ROOT}"

    echo "[launch] model=${model_tag} cuda=${cuda_devices} prompt=${prompt_type} baseline=${baseline} trunc=${truncation_ratio} setting=${setting_tag}"
    echo "[output] ${output_dir}"
    echo "[seeds] ${seeds_csv}"
    echo "[run_order] ${RUN_ORDER}"

    IFS=',' read -r -a seed_items <<< "${seeds_csv}"
    if [[ "${RUN_ORDER}" == "benchmark_major" ]]; then
        for data_name in "${DATA_NAME_ITEMS[@]}"; do
            for seed in "${seed_items[@]}"; do
                if [[ ${#seed_items[@]} -gt 1 ]]; then
                    log_suffix="_${data_name}_seed${seed}"
                else
                    log_suffix="_${data_name}"
                fi

                log_file="${LOG_ROOT}/${model_tag}_${baseline}_${tr_tag}_${setting_tag}${log_suffix}.log"

                echo "[log] ${log_file}"

                (
                    local -a cmd

                    cd "${EVAL_DIR}"
                    cmd=(
                        python3 -u math_eval.py
                        --model_name_or_path "${model_path}" \
                        --data_names "${data_name}" \
                        --output_dir "${output_dir}" \
                        --split "${SPLIT}" \
                        --prompt_type "${prompt_type}" \
                        --apply_chat_template \
                        --cot_baseline "${baseline}" \
                        --lc_ratio "${LC_RATIO}" \
                        --truncation_ratio "${truncation_ratio}" \
                        --num_test_sample "${NUM_TEST_SAMPLE}" \
                        --seed "${seed}" \
                        --temperature "${temperature}" \
                        --n_sampling "${n_sampling}" \
                        --top_p 1 \
                        --start 0 \
                        --end -1 \
                        --use_vllm \
                        --save_outputs \
                        --pipeline_parallel_size "${PIPELINE_PARALLEL_SIZE}" \
                        --max_tokens_per_call "${MAX_TOKENS}"
                    )

                    echo "[info] data_name=${data_name}"
                    echo "[info] seed=${seed}"
                    if [[ "${OVERWRITE}" == "1" ]]; then
                        echo "[info] overwrite enabled"
                        cmd+=(--overwrite)
                    else
                        echo "[info] overwrite disabled, will resume existing outputs when possible"
                    fi

                    TOKENIZERS_PARALLELISM=false \
                    CUDA_VISIBLE_DEVICES="${cuda_devices}" \
                    "${cmd[@]}"
                ) >"${log_file}" 2>&1
            done

            run_repeat_summary "${output_dir}" "${data_name}" "${temperature}" "${n_sampling}" "${model_path}" "${seeds_csv}" "${prompt_type}"
        done
    else
        for seed in "${seed_items[@]}"; do
            if [[ ${#seed_items[@]} -gt 1 ]]; then
                log_suffix="_seed${seed}"
            else
                log_suffix=""
            fi

            log_file="${LOG_ROOT}/${model_tag}_${baseline}_${tr_tag}_${setting_tag}${log_suffix}.log"

            echo "[log] ${log_file}"

            (
                local -a cmd

                cd "${EVAL_DIR}"
                cmd=(
                    python3 -u math_eval.py
                    --model_name_or_path "${model_path}" \
                    --data_names "${DATA_NAMES}" \
                    --output_dir "${output_dir}" \
                    --split "${SPLIT}" \
                    --prompt_type "${prompt_type}" \
                    --apply_chat_template \
                    --cot_baseline "${baseline}" \
                    --lc_ratio "${LC_RATIO}" \
                    --truncation_ratio "${truncation_ratio}" \
                    --num_test_sample "${NUM_TEST_SAMPLE}" \
                    --seed "${seed}" \
                    --temperature "${temperature}" \
                    --n_sampling "${n_sampling}" \
                    --top_p 1 \
                    --start 0 \
                    --end -1 \
                    --use_vllm \
                    --save_outputs \
                    --pipeline_parallel_size "${PIPELINE_PARALLEL_SIZE}" \
                    --max_tokens_per_call "${MAX_TOKENS}"
                )

                echo "[info] seed=${seed}"
                if [[ "${OVERWRITE}" == "1" ]]; then
                    echo "[info] overwrite enabled"
                    cmd+=(--overwrite)
                else
                    echo "[info] overwrite disabled, will resume existing outputs when possible"
                fi

                TOKENIZERS_PARALLELISM=false \
                CUDA_VISIBLE_DEVICES="${cuda_devices}" \
                "${cmd[@]}"
            ) >"${log_file}" 2>&1
        done

        run_repeat_summary "${output_dir}" "${DATA_NAMES}" "${temperature}" "${n_sampling}" "${model_path}" "${seeds_csv}" "${prompt_type}"
    fi
}

run_model_suite() {
    local model_path=$1
    local cuda_devices=$2
    local prompt_type=$3
    local job
    local temperature
    local n_sampling
    local seeds_csv
    local rest
    local baseline
    local truncation_ratio

    for baseline in "${BASELINE_ITEMS[@]}"; do
        for truncation_ratio in "${TRUNCATION_RATIO_ITEMS[@]}"; do
            for job in "${JOB_ITEMS[@]}"; do
                temperature=${job%%:*}
                rest=${job#*:}
                n_sampling=${rest%%:*}
                seeds_csv=${rest#*:}
                run_single_setting "${model_path}" "${cuda_devices}" "${prompt_type}" \
                    "${baseline}" "${truncation_ratio}" \
                    "${temperature}" "${n_sampling}" "${seeds_csv}"
            done
        done
    done
}

if [[ -n "${SETTING_ALIAS}" ]]; then
    JOB_SPECS=$(resolve_setting_alias "${SETTING_ALIAS}")
else
    JOB_SPECS=""
    IFS=',' read -r -a SETTING_ITEMS <<< "${SETTING_SPECS}"
    for setting in "${SETTING_ITEMS[@]}"; do
        if [[ -n "${JOB_SPECS}" ]]; then
            JOB_SPECS="${JOB_SPECS};"
        fi
        JOB_SPECS="${JOB_SPECS}${setting}:${SEEDS}"
    done
fi
IFS=';' read -r -a JOB_ITEMS <<< "${JOB_SPECS}"
IFS=',' read -r -a DATA_NAME_ITEMS <<< "${DATA_NAMES}"
IFS=',' read -r -a BASELINE_ITEMS <<< "${BASELINES}"
IFS=',' read -r -a TRUNCATION_RATIO_ITEMS <<< "${TRUNCATION_RATIOS}"

echo "[info] prompt_type_1=${PROMPT_TYPE_1}"
echo "[info] prompt_type_2=${PROMPT_TYPE_2}"
echo "[info] data_names=${DATA_NAMES}"
echo "[info] baselines=${BASELINES}"
echo "[info] truncation_ratios=${TRUNCATION_RATIOS}"
echo "[info] lc_ratio=${LC_RATIO}"
echo "[info] setting_alias=${SETTING_ALIAS}"
echo "[info] setting_specs=${SETTING_SPECS}"
echo "[info] seeds=${SEEDS}"
echo "[info] job_specs=${JOB_SPECS}"
echo "[info] run_order=${RUN_ORDER}"
echo "[info] model_1=${MODEL_1} cuda=${CUDA_DEVICES_1}"
echo "[info] model_2=${MODEL_2} cuda=${CUDA_DEVICES_2}"
echo "[info] max_tokens=${MAX_TOKENS} pipeline_parallel_size=${PIPELINE_PARALLEL_SIZE} overwrite=${OVERWRITE}"

if [[ "${CUDA_DEVICES_1}" == "${CUDA_DEVICES_2}" ]]; then
    echo "[info] single GPU mode detected, running all jobs sequentially"
    run_model_suite "${MODEL_1}" "${CUDA_DEVICES_1}" "${PROMPT_TYPE_1}"
    run_model_suite "${MODEL_2}" "${CUDA_DEVICES_2}" "${PROMPT_TYPE_2}"
else
    echo "[info] multi GPU mode detected, running the two model suites in parallel"

    run_model_suite "${MODEL_1}" "${CUDA_DEVICES_1}" "${PROMPT_TYPE_1}" &
    PID_1=$!

    run_model_suite "${MODEL_2}" "${CUDA_DEVICES_2}" "${PROMPT_TYPE_2}" &
    PID_2=$!

    wait "${PID_1}"
    wait "${PID_2}"
fi

echo "[done] all runs finished"
