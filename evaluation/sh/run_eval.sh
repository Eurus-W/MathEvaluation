cd /nfsdata/whq/projects/Qwen2.5-Math/evaluation
bash sh/run_registered_model_naive_baselines.sh qwen3_4b "0,1" &
bash sh/run_registered_model_naive_baselines.sh deepseek_qwen7b 4 &
bash sh/run_registered_model_naive_baselines.sh deepseek_llama8b 5 &
