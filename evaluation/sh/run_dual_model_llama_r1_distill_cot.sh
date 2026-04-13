#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ $# -lt 2 ]]; then
    cat <<'EOF'
Usage:
  bash sh/run_dual_model_llama_r1_distill_cot.sh <model_1> <model_2> [cuda_devices_model_1] [cuda_devices_model_2]

Example:
  bash sh/run_dual_model_llama_r1_distill_cot.sh \
      /path/to/model_a \
      /path/to/model_b \
      0 \
      1

Notes:
  - The script fixes prompt_type=llama-r1-distill-cot.
  - Default benchmarks: math,minerva_math,olympiadbench,aime24.
  - Default decoding settings:
      1) temperature=0.0, n_sampling=1
      2) temperature=0.6, n_sampling=1
      3) temperature=1.0, n_sampling=16
  - If the two CUDA device arguments are different, the two models run in parallel.
  - If the two CUDA device arguments are the same, all jobs run sequentially on that single GPU.
  - Optional env overrides:
      DATA_NAMES="math,minerva_math,olympiadbench,aime24,amc23"
      SETTING_SPECS="0.0:1,0.6:1"
      SEEDS="0,1,2,3,4"
      SETTING_ALIAS="all"
      OVERWRITE=0
EOF
    exit 1
fi

MODEL_1=$1
MODEL_2=$2
CUDA_DEVICES_1=${3:-0}
CUDA_DEVICES_2=${4:-1}

PROMPT_TYPE="llama-r1-distill-cot"
DATA_NAMES=${DATA_NAMES:-"math,minerva_math,olympiadbench,aime24"}
SETTING_SPECS=${SETTING_SPECS:-"0.0:1,0.6:1,1.0:16"}
SEEDS=${SEEDS:-"0"}
SETTING_ALIAS=${SETTING_ALIAS:-""}
SPLIT="test"
NUM_TEST_SAMPLE=${NUM_TEST_SAMPLE:--1}
MAX_TOKENS=${MAX_TOKENS:-2048}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
OVERWRITE=${OVERWRITE:-1}

RUN_ROOT="${EVAL_DIR}/outputs/dual_model_llama_r1_distill_cot"
LOG_ROOT="${EVAL_DIR}/logs/dual_model_llama_r1_distill_cot"
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

print_setting_line() {
    local index=$1
    local model_path=$2
    local cuda_devices=$3
    local temperature=$4
    local n_sampling=$5
    local seeds_csv=$6

    local model_tag
    model_tag=$(sanitize_name "${model_path}")

    printf "  %d. model=%s | cuda=%s | prompt=%s | data_names=%s | temperature=%s | n_sampling=%s | seeds=%s | max_tokens=%s\n" \
        "${index}" \
        "${model_tag}" \
        "${cuda_devices}" \
        "${PROMPT_TYPE}" \
        "${DATA_NAMES}" \
        "${temperature}" \
        "${n_sampling}" \
        "${seeds_csv}" \
        "${MAX_TOKENS}"
}

run_repeat_summary() {
    local output_dir=$1
    local temperature=$2
    local n_sampling=$3
    local model_path=$4
    local seeds_csv=$5
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
            --data_names "${DATA_NAMES}" \
            --prompt_type "${PROMPT_TYPE}" \
            --temperature "${temperature}" \
            --n_sampling "${n_sampling}" \
            --seeds "${seeds_csv}" \
            --model_name_or_path "${model_path}"
    )
}

print_execution_plan() {
    local index
    local job
    local temperature
    local n_sampling
    local seeds_csv

    echo "[plan] upcoming jobs:"

    if [[ "${CUDA_DEVICES_1}" == "${CUDA_DEVICES_2}" ]]; then
        echo "[plan] mode=single_gpu_sequential"
        index=1
        for job in "${JOB_ITEMS[@]}"; do
            temperature=${job%%:*}
            local rest=${job#*:}
            n_sampling=${rest%%:*}
            seeds_csv=${rest#*:}
            print_setting_line "${index}" "${MODEL_1}" "${CUDA_DEVICES_1}" "${temperature}" "${n_sampling}" "${seeds_csv}"
            index=$((index + 1))
        done
        for job in "${JOB_ITEMS[@]}"; do
            temperature=${job%%:*}
            local rest=${job#*:}
            n_sampling=${rest%%:*}
            seeds_csv=${rest#*:}
            print_setting_line "${index}" "${MODEL_2}" "${CUDA_DEVICES_2}" "${temperature}" "${n_sampling}" "${seeds_csv}"
            index=$((index + 1))
        done
    else
        echo "[plan] mode=multi_gpu_parallel"
        echo "[plan] suite on cuda=${CUDA_DEVICES_1}:"
        index=1
        for job in "${JOB_ITEMS[@]}"; do
            temperature=${job%%:*}
            local rest=${job#*:}
            n_sampling=${rest%%:*}
            seeds_csv=${rest#*:}
            print_setting_line "${index}" "${MODEL_1}" "${CUDA_DEVICES_1}" "${temperature}" "${n_sampling}" "${seeds_csv}"
            index=$((index + 1))
        done
        echo "[plan] suite on cuda=${CUDA_DEVICES_2}:"
        index=1
        for job in "${JOB_ITEMS[@]}"; do
            temperature=${job%%:*}
            local rest=${job#*:}
            n_sampling=${rest%%:*}
            seeds_csv=${rest#*:}
            print_setting_line "${index}" "${MODEL_2}" "${CUDA_DEVICES_2}" "${temperature}" "${n_sampling}" "${seeds_csv}"
            index=$((index + 1))
        done
    fi
}

run_single_setting() {
    local model_path=$1
    local cuda_devices=$2
    local temperature=$3
    local n_sampling=$4
    local seeds_csv=$5
    local seed
    local log_file
    local output_dir
    local log_suffix
    local -a seed_items

    local model_tag
    model_tag=$(sanitize_name "${model_path}")

    local setting_tag="t${temperature}_n${n_sampling}"
    output_dir="${RUN_ROOT}/${model_tag}/${setting_tag}"

    mkdir -p "${output_dir}" "${LOG_ROOT}"

    echo "[launch] model=${model_tag} cuda=${cuda_devices} setting=${setting_tag}"
    echo "[output] ${output_dir}"
    echo "[seeds] ${seeds_csv}"

    IFS=',' read -r -a seed_items <<< "${seeds_csv}"
    for seed in "${seed_items[@]}"; do
        if [[ ${#seed_items[@]} -gt 1 ]]; then
            log_suffix="_seed${seed}"
        else
            log_suffix=""
        fi

        log_file="${LOG_ROOT}/${model_tag}_${setting_tag}${log_suffix}.log"

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
                --prompt_type "${PROMPT_TYPE}" \
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

    run_repeat_summary "${output_dir}" "${temperature}" "${n_sampling}" "${model_path}" "${seeds_csv}"
}

run_model_suite() {
    local model_path=$1
    local cuda_devices=$2
    local job
    local temperature
    local n_sampling
    local seeds_csv
    local rest

    for job in "${JOB_ITEMS[@]}"; do
        temperature=${job%%:*}
        rest=${job#*:}
        n_sampling=${rest%%:*}
        seeds_csv=${rest#*:}
        run_single_setting "${model_path}" "${cuda_devices}" "${temperature}" "${n_sampling}" "${seeds_csv}"
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

echo "[info] prompt_type=${PROMPT_TYPE}"
echo "[info] data_names=${DATA_NAMES}"
echo "[info] setting_alias=${SETTING_ALIAS}"
echo "[info] setting_specs=${SETTING_SPECS}"
echo "[info] seeds=${SEEDS}"
echo "[info] job_specs=${JOB_SPECS}"
echo "[info] model_1=${MODEL_1} cuda=${CUDA_DEVICES_1}"
echo "[info] model_2=${MODEL_2} cuda=${CUDA_DEVICES_2}"
echo "[info] max_tokens=${MAX_TOKENS} pipeline_parallel_size=${PIPELINE_PARALLEL_SIZE} overwrite=${OVERWRITE}"
print_execution_plan

if [[ "${CUDA_DEVICES_1}" == "${CUDA_DEVICES_2}" ]]; then
    echo "[info] single GPU mode detected, running all jobs sequentially"
    run_model_suite "${MODEL_1}" "${CUDA_DEVICES_1}"
    run_model_suite "${MODEL_2}" "${CUDA_DEVICES_2}"
else
    echo "[info] multi GPU mode detected, running the two model suites in parallel"

    run_model_suite "${MODEL_1}" "${CUDA_DEVICES_1}" &
    PID_1=$!

    run_model_suite "${MODEL_2}" "${CUDA_DEVICES_2}" &
    PID_2=$!

    wait "${PID_1}"
    wait "${PID_2}"
fi

echo "[done] all runs finished"
