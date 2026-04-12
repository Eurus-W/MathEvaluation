mkdir -p logs

CUDA_VISIBLE_DEVICES="0" bash sh/eval.sh llama-r1-distill-cot /nfsdata/whq/models/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B > logs/llama-r1-distill-cot-greedy.log 2>&1 &

CUDA_VISIBLE_DEVICES="1" bash sh/eval.sh llama-r1-distill /nfsdata/whq/models/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B > logs/llama-r1-distill-greedy.log 2>&1 &

CUDA_VISIBLE_DEVICES="2" bash sh/eval.sh llama-r1-distill-cot /nfsdata/whq/models/models/short_models > logs/llama-r1-distill-short-cot-greedy.log 2>&1 &

CUDA_VISIBLE_DEVICES="3" bash sh/eval.sh llama-r1-distill /nfsdata/whq/models/models/short_models > logs/llama-r1-distill-short-greedy.log 2>&1 &


# CUDA_VISIBLE_DEVICES="3" bash sh/eval.sh llama-r1-distill-cot /nfsdata/whq/models/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B 512 > logs/llama-r1-distill-cot-greedy-512.log 2>&1 &

# CUDA_VISIBLE_DEVICES="4" bash sh/eval.sh llama-r1-distill /nfsdata/whq/models/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B 512 > logs/llama-r1-distill-greedy-512.log 2>&1 &

# CUDA_VISIBLE_DEVICES="5" bash sh/eval.sh llama-r1-distill-cot /nfsdata/whq/models/models/short_models 512 > logs/llama-r1-distill-short-cot-greedy-512.log 2>&1 &

# CUDA_VISIBLE_DEVICES="7" bash sh/eval.sh llama-r1-distill /nfsdata/whq/models/models/short_models 512 > logs/llama-r1-distill-short-greedy-512.log 2>&1 &
