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

用于同时评测两个模型，固定使用 `llama-r1-distill-cot` prompt，并按多组采样配置自动运行。

## 用途

这个脚本适合做下面几类工作：

- 对比两个模型在同一批 benchmark 上的表现
- 一次性跑完多组 decoding 配置
- 在两张 GPU 上并行跑两个模型，或者在单张 GPU 上串行跑完

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

## 常用示例

1. 两个模型分别占一张卡并行跑默认 benchmark

```bash
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  0 \
  1
```

2. 只跑两组设置

```bash
SETTING_SPECS="0.0:1,0.6:1" \
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  0 \
  1
```

3. 加入 `math_500`

```bash
DATA_NAMES="math,math_500,minerva_math,olympiadbench,aime24" \
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  0 \
  1
```

4. 单卡串行跑

```bash
bash sh/run_dual_model_llama_r1_distill_cot.sh \
  /path/to/model_a \
  /path/to/model_b \
  0 \
  0
```

## 可选环境变量

运行前可以通过环境变量覆盖默认配置：

- `DATA_NAMES`
- `SETTING_SPECS`
- `MAX_TOKENS`
- `NUM_TEST_SAMPLE`
- `PIPELINE_PARALLEL_SIZE`
- `OVERWRITE`

示例：

```bash
DATA_NAMES="math,math_500" \
SETTING_SPECS="0.0:1" \
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

其中：

- `<model_tag>` 来自模型路径最后两级目录，并把 `/` 替换成 `_`
- `<setting_tag>` 形如 `t0.0_n1`

## 运行前注意

- 需要先安装 `evaluation/requirements.txt` 里的依赖
- 默认使用 `vllm` 跑推理
- 脚本会进入 `evaluation/` 目录后再执行 `math_eval.py`
- 当 `OVERWRITE=1` 时，会覆盖已有同名结果
- 当 `OVERWRITE=0` 时，会尽量复用已有输出继续跑

## 结果检查

脚本运行时会先打印执行计划，包括：

- 每个模型使用的 GPU
- 每组 temperature / n_sampling
- 当前跑的 benchmark 列表
- 输出目录和日志路径

全部完成后会打印：

```text
[done] all runs finished
```


### Evaluation
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
