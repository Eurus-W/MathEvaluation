### Requirements
You can install the required packages with the following command:
```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3
```

# `run_dual_model_llama_r1_distill_cot.sh`

用于同时评测两个模型，固定使用 `llama-r1-distill-cot` prompt，并按多组 decoding 配置自动运行。

## 用途

这个脚本适合做下面几类工作：

- 对比两个模型在同一批 benchmark 上的表现
- 一次性跑完多组 decoding 配置
- 用单张 GPU 串行跑完两个模型，或者用两张 GPU 并行跑
- 对 `temperature=0.6, n_sampling=1` 做多次重复实验并自动汇总 5 次结果

## 基本用法

在 `Qwen2.5-Math/evaluation/` 目录下执行：

```bash
bash sh/run_dual_model_llama_r1_distill_cot.sh <model_1> <model_2> [cuda_devices_model_1] [cuda_devices_model_2]
```

示例：

```bash
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  0 \
  1
```

如果最后两个 GPU 参数不同，两个模型会并行跑。

如果最后两个 GPU 参数相同，所有任务会在这张卡上串行跑完。

## 默认配置

脚本内置的默认值如下：

- `PROMPT_TYPE=llama-r1-distill-cot`
- `DATA_NAMES=math,minerva_math,olympiadbench,aime24`
- `SETTING_SPECS=0.0:1,0.6:1,1.0:16`
- `SEEDS=0`
- `SETTING_ALIAS=""`
- `RUN_ORDER=seed_major`
- `MAX_TOKENS=2048`
- `NUM_TEST_SAMPLE=-1`
- `OVERWRITE=1`

其中 `SETTING_SPECS` 的格式是：

```text
temperature:n_sampling,temperature:n_sampling,...
```

例如：

- `0.0:1` 表示 greedy
- `1.0:16` 表示温度为 1.0，采样 16 次

如果你想让某个 setting 跑多个 seed，可以直接配合 `SEEDS` 使用。例如：

```bash
SETTING_SPECS="0.6:1" \
SEEDS="0,1,2,3,4" \
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  0 \
  1
```

`RUN_ORDER` 支持两种模式：

- `seed_major`
  先跑完一个 setting 下的所有 benchmark，再切下一个 seed
- `benchmark_major`
  先跑完一个 benchmark 的所有 seed，再切下一个 benchmark

如果你希望尽快看到单个 benchmark 的完整多 seed 结果，推荐使用 `RUN_ORDER="benchmark_major"`。

## `SETTING_ALIAS`

为了少配参数，脚本支持几个短别名：

- `SETTING_ALIAS="basic"`
  代表 `0.0:1:0;0.6:1:0;1.0:16:0`
- `SETTING_ALIAS="t06x5"`
  代表 `0.6:1:0,1,2,3,4`
- `SETTING_ALIAS="all"`
  代表 `0.0:1:0;0.6:1:0,1,2,3,4;1.0:16:0`

其中 `all` 的含义是：

- `temperature=0.0, n_sampling=1, seed=0`
- `temperature=0.6, n_sampling=1, seed=0,1,2,3,4`
- `temperature=1.0, n_sampling=16, seed=0`

这个别名适合做完整一轮实验，而且不会重复多跑一遍 `t=0.6, seed=0`。

当 `SETTING_ALIAS` 非空时，脚本会优先使用别名展开后的 job 列表，忽略手动传入的 `SETTING_SPECS` 和 `SEEDS` 组合。

## 常用示例

1. 两个模型分别占一张卡并行跑默认 benchmark

```bash
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  0 \
  1
```

2. 单卡串行跑

```bash
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  0 \
  0
```

3. 只跑两组设置

```bash
SETTING_SPECS="0.0:1,0.6:1" \
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  0 \
  1
```

4. 跑 5 个 benchmark，使用两张卡 `6/7`

```bash
DATA_NAMES="math,minerva_math,olympiadbench,aime24,amc23" \
MAX_TOKENS=16384 \
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  6 \
  7
```

5. 只跑新的 `temperature=0.6` 5 次重复实验

```bash
DATA_NAMES="math,minerva_math,olympiadbench,aime24,amc23" \
SETTING_ALIAS="t06x5" \
MAX_TOKENS=16384 \
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  6 \
  7
```

6. 一键跑完整实验

```bash
DATA_NAMES="math,minerva_math,olympiadbench,aime24,amc23" \
SETTING_ALIAS="all" \
MAX_TOKENS=16384 \
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  6 \
  7
```

7. 复用已经完成的 `seed0`，按 benchmark 依次补齐 `temperature=0.6` 的 5 次实验

```bash
DATA_NAMES="math,minerva_math,olympiadbench,aime24,amc23" \
SETTING_ALIAS="t06x5" \
RUN_ORDER="benchmark_major" \
MAX_TOKENS=16384 \
OVERWRITE=0 \
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  6 \
  7
```

## 可选环境变量

运行前可以通过环境变量覆盖默认配置：

- `DATA_NAMES`
- `SETTING_SPECS`
- `SEEDS`
- `SETTING_ALIAS`
- `RUN_ORDER`
- `MAX_TOKENS`
- `NUM_TEST_SAMPLE`
- `PIPELINE_PARALLEL_SIZE`
- `OVERWRITE`

示例：

```bash
DATA_NAMES="math,math_500" \
SETTING_SPECS="0.0:1" \
SEEDS="0,1,2,3,4" \
RUN_ORDER="benchmark_major" \
MAX_TOKENS=4096 \
OVERWRITE=0 \
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  0 \
  1
```

## 输出位置

结果和日志会分别保存在：

- 输出：`evaluation/outputs/dual_model_llama_r1_distill_cot/<model_tag>/<setting_tag>/`
- 日志：`evaluation/logs/dual_model_llama_r1_distill_cot/<model_tag>_<setting_tag>.log`
- 多 seed 日志：`evaluation/logs/dual_model_llama_r1_distill_cot/<model_tag>_<setting_tag>_seed<seed>.log`
- `benchmark_major` 模式下的日志：`evaluation/logs/dual_model_llama_r1_distill_cot/<model_tag>_<setting_tag>_<data_name>_seed<seed>.log`

其中：

- `<model_tag>` 来自模型路径最后两级目录，并把 `/` 替换成 `_`
- `<setting_tag>` 形如 `t0.0_n1`

当某个 setting 跑多个 seed 时，原始输出仍然写在同一个 `<setting_tag>/` 目录下，但文件名里会带 `seed`。

对于多 seed setting，脚本还会额外生成一个汇总文件：

- `evaluation/outputs/dual_model_llama_r1_distill_cot/<model_tag>/<setting_tag>/repeat_summary_t<temperature>_n<n_sampling>_seeds_<seed-list>.json`
- `benchmark_major` 模式下会按 benchmark 额外生成：
  `evaluation/outputs/dual_model_llama_r1_distill_cot/<model_tag>/<setting_tag>/repeat_summary_<data_name>_t<temperature>_n<n_sampling>_seeds_<seed-list>.json`

这个汇总文件会包含：

- 每个 benchmark 的 5 次 `acc`
- 每个 benchmark 的 5 次 `avg_output_tokens`
- 每个 benchmark 的平均 `acc`
- 每个 benchmark 的平均 `avg_output_tokens`
- overall 平均结果

## 运行前注意

- 需要先安装 `evaluation/requirements.txt` 里的依赖
- 默认使用 `vllm` 跑推理
- 脚本会进入 `evaluation/` 目录后再执行 `math_eval.py`
- 当 `OVERWRITE=1` 时，会覆盖已有同名结果
- 当 `OVERWRITE=0` 时，会尽量复用已有输出继续跑
- 如果你已经确认当前 `seed0` 是正确的正式结果，做多 seed 补跑时通常推荐 `OVERWRITE=0`

## 结果检查

脚本运行时会先打印执行计划，包括：

- 每个模型使用的 GPU
- 每组 temperature / n_sampling / seeds
- 当前跑的 benchmark 列表
- 输出目录和日志路径

全部完成后会打印：

```text
[done] all runs finished
```


### Evaluation
This section documents the original single-model entrypoints and should be treated as legacy usage.

- If you are running the current `llama-r1-distill-cot` comparison workflow, prefer `sh/run_dual_model_llama_r1_distill_cot.sh`.
- Use the commands below mainly for the older Qwen2.5/Qwen2-Math-Instruct evaluation flow.

You can evaluate Qwen2.5/Qwen2-Math-Instruct series model with the following command:
```bash
# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen25-math-cot"
# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2.5-Math-7B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2.5-Math-72B-Instruct
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-72B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH


# Qwen2-Math-Instruct Series
PROMPT_TYPE="qwen-boxed"
# Qwen2-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-1.5B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2-Math-7B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-7B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2-Math-72B-Instruct
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-72B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
```

## Acknowledgement
The codebase is adapted from [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).
