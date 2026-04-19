#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ $# -lt 2 ]]; then
    cat <<'EOF'
Usage:
  bash sh/run_registered_model_default_eval.sh <model_id> <cuda_devices>

Registered model ids:
  deepseek_qwen7b
  deepseek_llama8b
  qwen3_4b

Examples:
  bash sh/run_registered_model_default_eval.sh deepseek_qwen7b 4
  bash sh/run_registered_model_default_eval.sh deepseek_llama8b 5
  bash sh/run_registered_model_default_eval.sh qwen3_4b 6

Default behavior:
  - DATA_NAMES="gsm8k,math_500,minerva_math,aime24,aime25,aime26,amc23"
  - SETTING_ALIAS="t06x5"
  - RUN_ORDER="benchmark_major"
  - NUM_TEST_SAMPLE=-1
  - TOP_P=0.95
  - model-specific top_k / max_tokens:
      deepseek_qwen7b  -> top_k disabled, max_tokens=16384
      deepseek_llama8b -> top_k disabled, max_tokens=16384
      qwen3_4b         -> top_k=20,     max_tokens=81920

Optional env overrides:
  DATA_NAMES="gsm8k,math_500,minerva_math,aime24,aime25,aime26,amc23"
  SETTING_ALIAS="t06x5"
  SETTING_SPECS="0.6:1"
  SEEDS="0,1,2,3,4"
  RUN_ORDER="benchmark_major"
  NUM_TEST_SAMPLE=-1
  TOP_P=0.95
  OVERWRITE=0
  PIPELINE_PARALLEL_SIZE=1
  DRY_RUN=1
  PROMPT_TYPE_OVERRIDE="qwen25-math-cot"
  MAX_TOKENS_OVERRIDE=16384
  TOP_K_OVERRIDE=20
EOF
    exit 1
fi

MODEL_ID=$1
CUDA_DEVICES=$2

sanitize_name() {
    local value=$1
    value=${value%/}
    value=$(echo "${value}" | sed 's#/*$##' | awk -F/ '{if (NF >= 2) print $(NF-1) "/" $NF; else print $NF}')
    value=${value// /_}
    value=${value//\//_}
    echo "${value}"
}

resolve_model_config() {
    local model_id=$1
    case "${model_id}" in
        deepseek_qwen7b|r1_qwen7b|qwen_distill_7b|/nfsdata/whq/models/models/Qwen/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
            RESOLVED_MODEL_NAME_OR_PATH="/nfsdata/whq/models/models/Qwen/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            RESOLVED_PROMPT_TYPE="qwen25-math-cot"
            RESOLVED_MAX_TOKENS=16384
            RESOLVED_TOP_K=-1
            ;;
        deepseek_llama8b|r1_llama8b|llama_distill_8b|/nfsdata/whq/models/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
            RESOLVED_MODEL_NAME_OR_PATH="/nfsdata/whq/models/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
            RESOLVED_PROMPT_TYPE="llama-r1-distill-cot"
            RESOLVED_MAX_TOKENS=16384
            RESOLVED_TOP_K=-1
            ;;
        qwen3_4b|qwen3_4b_thinking|qwen3_thinking_4b|/nfsdata/whq/models/models/Qwen/Qwen3-4B-Thinking-2507)
            RESOLVED_MODEL_NAME_OR_PATH="/nfsdata/whq/models/models/Qwen/Qwen3-4B-Thinking-2507"
            RESOLVED_PROMPT_TYPE="qwen25-math-cot"
            RESOLVED_MAX_TOKENS=81920
            RESOLVED_TOP_K=20
            ;;
        *)
            echo "[error] unknown model_id=${model_id}" >&2
            echo "[error] supported model ids: deepseek_qwen7b, deepseek_llama8b, qwen3_4b" >&2
            exit 1
            ;;
    esac
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

build_setting_tag() {
    local temperature=$1
    local n_sampling=$2
    local top_p=$3
    local top_k=$4

    local tag="t${temperature}_p${top_p}_n${n_sampling}"
    if [[ ${top_k} -gt 0 ]]; then
        tag="${tag}_k${top_k}"
    fi
    echo "${tag}"
}

write_run_manifest() {
    local output_dir=$1
    local setting_tag=$2

    mkdir -p "${output_dir}"
    cat > "${output_dir}/run_manifest.json" <<EOF
{
  "model_id": "${MODEL_ID}",
  "model_name_or_path": "${MODEL_NAME_OR_PATH}",
  "prompt_type": "${PROMPT_TYPE}",
  "data_names": "${DATA_NAMES}",
  "setting_alias": "${SETTING_ALIAS}",
  "setting_specs": "${SETTING_SPECS}",
  "seeds": "${SEEDS}",
  "job_specs": "${JOB_SPECS}",
  "run_order": "${RUN_ORDER}",
  "num_test_sample": ${NUM_TEST_SAMPLE},
  "top_p": ${TOP_P},
  "top_k": ${TOP_K},
  "max_tokens": ${MAX_TOKENS},
  "pipeline_parallel_size": ${PIPELINE_PARALLEL_SIZE},
  "overwrite": ${OVERWRITE},
  "cuda_devices": "${CUDA_DEVICES}",
  "setting_tag": "${setting_tag}"
}
EOF
}

print_dry_run_command() {
    local cuda_devices=$1
    shift
    printf '[dry_run] cmd=TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=%q ' "${cuda_devices}"
    printf '%q ' "$@"
    printf '\n'
}

run_repeat_summary() {
    local output_dir=$1
    local data_names_csv=$2
    local temperature=$3
    local n_sampling=$4
    local model_path=$5
    local prompt_type=$6
    local seeds_csv=$7
    local seed_count

    IFS=',' read -r -a seed_items <<< "${seeds_csv}"
    seed_count=${#seed_items[@]}

    if [[ ${seed_count} -le 1 ]]; then
        return 0
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry_run] skip summarize_repeat_metrics output_dir=${output_dir} temperature=${temperature} n_sampling=${n_sampling} seeds=${seeds_csv}"
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
    local temperature=$1
    local n_sampling=$2
    local seeds_csv=$3
    local seed
    local log_file
    local output_dir
    local log_dir
    local log_suffix
    local -a seed_items
    local data_name
    local model_tag
    local setting_tag

    model_tag=$(sanitize_name "${MODEL_NAME_OR_PATH}")
    setting_tag=$(build_setting_tag "${temperature}" "${n_sampling}" "${TOP_P}" "${TOP_K}")
    output_dir="${RUN_ROOT}/models/${model_tag}/settings/${setting_tag}"
    log_dir="${LOG_ROOT}/models/${model_tag}/settings/${setting_tag}"

    if [[ "${DRY_RUN}" != "1" ]]; then
        mkdir -p "${output_dir}" "${log_dir}"
        write_run_manifest "${output_dir}" "${setting_tag}"
    fi

    echo "[launch] model=${model_tag} cuda=${CUDA_DEVICES} prompt=${PROMPT_TYPE} setting=${setting_tag}"
    echo "[output] ${output_dir}"
    echo "[log_dir] ${log_dir}"
    echo "[seeds] ${seeds_csv}"
    echo "[run_order] ${RUN_ORDER}"

    IFS=',' read -r -a seed_items <<< "${seeds_csv}"
    if [[ "${RUN_ORDER}" == "benchmark_major" ]]; then
        for data_name in "${DATA_NAME_ITEMS[@]}"; do
            for seed in "${seed_items[@]}"; do
                if [[ ${#seed_items[@]} -gt 1 ]]; then
                    log_suffix="data_${data_name}_seed${seed}"
                else
                    log_suffix="data_${data_name}"
                fi

                log_file="${log_dir}/${log_suffix}.log"
                echo "[log] ${log_file}"

                local -a cmd
                cmd=(
                    python3 -u math_eval.py
                    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
                    --data_names "${data_name}" \
                    --output_dir "${output_dir}" \
                    --split test \
                    --prompt_type "${PROMPT_TYPE}" \
                    --num_test_sample "${NUM_TEST_SAMPLE}" \
                    --seed "${seed}" \
                    --temperature "${temperature}" \
                    --n_sampling "${n_sampling}" \
                    --top_p "${TOP_P}" \
                    --top_k "${TOP_K}" \
                    --start 0 \
                    --end -1 \
                    --use_vllm \
                    --apply_chat_template \
                    --save_outputs \
                    --pipeline_parallel_size "${PIPELINE_PARALLEL_SIZE}" \
                    --max_tokens_per_call "${MAX_TOKENS}"
                )

                if [[ "${OVERWRITE}" == "1" ]]; then
                    cmd+=(--overwrite)
                fi

                if [[ "${DRY_RUN}" == "1" ]]; then
                    echo "[dry_run] data_name=${data_name}"
                    echo "[dry_run] seed=${seed}"
                    print_dry_run_command "${CUDA_DEVICES}" "${cmd[@]}"
                else
                    (
                        cd "${EVAL_DIR}"
                        echo "[info] data_name=${data_name}"
                        echo "[info] seed=${seed}"
                        if [[ "${OVERWRITE}" == "1" ]]; then
                            echo "[info] overwrite enabled"
                        else
                            echo "[info] overwrite disabled, will resume existing outputs when possible"
                        fi

                        TOKENIZERS_PARALLELISM=false \
                        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
                        "${cmd[@]}"
                    ) >"${log_file}" 2>&1
                fi
            done

            run_repeat_summary "${output_dir}" "${data_name}" "${temperature}" "${n_sampling}" "${MODEL_NAME_OR_PATH}" "${PROMPT_TYPE}" "${seeds_csv}"
        done
    else
        for seed in "${seed_items[@]}"; do
            if [[ ${#seed_items[@]} -gt 1 ]]; then
                log_suffix="seed${seed}"
            else
                log_suffix="run"
            fi

            log_file="${log_dir}/${log_suffix}.log"
            echo "[log] ${log_file}"

            local -a cmd
            cmd=(
                python3 -u math_eval.py
                --model_name_or_path "${MODEL_NAME_OR_PATH}" \
                --data_names "${DATA_NAMES}" \
                --output_dir "${output_dir}" \
                --split test \
                --prompt_type "${PROMPT_TYPE}" \
                --num_test_sample "${NUM_TEST_SAMPLE}" \
                --seed "${seed}" \
                --temperature "${temperature}" \
                --n_sampling "${n_sampling}" \
                --top_p "${TOP_P}" \
                --top_k "${TOP_K}" \
                --start 0 \
                --end -1 \
                --use_vllm \
                --apply_chat_template \
                --save_outputs \
                --pipeline_parallel_size "${PIPELINE_PARALLEL_SIZE}" \
                --max_tokens_per_call "${MAX_TOKENS}"
            )

            if [[ "${OVERWRITE}" == "1" ]]; then
                cmd+=(--overwrite)
            fi

            if [[ "${DRY_RUN}" == "1" ]]; then
                echo "[dry_run] seed=${seed}"
                print_dry_run_command "${CUDA_DEVICES}" "${cmd[@]}"
            else
                (
                    cd "${EVAL_DIR}"
                    echo "[info] seed=${seed}"
                    if [[ "${OVERWRITE}" == "1" ]]; then
                        echo "[info] overwrite enabled"
                    else
                        echo "[info] overwrite disabled, will resume existing outputs when possible"
                    fi

                    TOKENIZERS_PARALLELISM=false \
                    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
                    "${cmd[@]}"
                ) >"${log_file}" 2>&1
            fi
        done

        run_repeat_summary "${output_dir}" "${DATA_NAMES}" "${temperature}" "${n_sampling}" "${MODEL_NAME_OR_PATH}" "${PROMPT_TYPE}" "${seeds_csv}"
    fi
}

run_model_suite() {
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
        run_single_setting "${temperature}" "${n_sampling}" "${seeds_csv}"
    done
}

resolve_model_config "${MODEL_ID}"

MODEL_NAME_OR_PATH="${RESOLVED_MODEL_NAME_OR_PATH}"
PROMPT_TYPE="${PROMPT_TYPE_OVERRIDE:-${RESOLVED_PROMPT_TYPE}}"
MAX_TOKENS="${MAX_TOKENS_OVERRIDE:-${RESOLVED_MAX_TOKENS}}"
TOP_K="${TOP_K_OVERRIDE:-${RESOLVED_TOP_K}}"
DATA_NAMES="${DATA_NAMES:-gsm8k,math_500,minerva_math,aime24,aime25,aime26,amc23}"
SETTING_SPECS="${SETTING_SPECS:-0.6:1}"
SEEDS="${SEEDS:-0,1,2,3,4}"
SETTING_ALIAS="${SETTING_ALIAS:-t06x5}"
RUN_ORDER="${RUN_ORDER:-benchmark_major}"
NUM_TEST_SAMPLE="${NUM_TEST_SAMPLE:--1}"
TOP_P="${TOP_P:-0.95}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_ROOT="${EVAL_DIR}/outputs/registered_model_default_eval"
LOG_ROOT="${EVAL_DIR}/logs/registered_model_default_eval"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

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

echo "[info] model_id=${MODEL_ID}"
echo "[info] model_name_or_path=${MODEL_NAME_OR_PATH}"
echo "[info] prompt_type=${PROMPT_TYPE}"
echo "[info] data_names=${DATA_NAMES}"
echo "[info] setting_alias=${SETTING_ALIAS}"
echo "[info] setting_specs=${SETTING_SPECS}"
echo "[info] seeds=${SEEDS}"
echo "[info] job_specs=${JOB_SPECS}"
echo "[info] run_order=${RUN_ORDER}"
echo "[info] top_p=${TOP_P}"
echo "[info] top_k=${TOP_K}"
echo "[info] max_tokens=${MAX_TOKENS}"
echo "[info] num_test_sample=${NUM_TEST_SAMPLE}"
echo "[info] pipeline_parallel_size=${PIPELINE_PARALLEL_SIZE}"
echo "[info] overwrite=${OVERWRITE}"
echo "[info] dry_run=${DRY_RUN}"
echo "[info] cuda_devices=${CUDA_DEVICES}"
echo "[info] outputs_root=${RUN_ROOT}"
echo "[info] logs_root=${LOG_ROOT}"

run_model_suite

echo "[done] all runs finished"
