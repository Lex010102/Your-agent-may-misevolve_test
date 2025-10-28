
models=(
    Qwen/Qwen2.5-7B
    andrewzh2/Absolute_Zero_Reasoner-Base-7b
    Qwen/Qwen2.5-Coder-7B
    andrewzh/Absolute_Zero_Reasoner-Coder-7b
    Qwen/Qwen2.5-14B
    andrewzh2/Absolute_Zero_Reasoner-Base-14b
    Qwen/Qwen2.5-Coder-14B
    andrewzh/Absolute_Zero_Reasoner-Coder-14b
)

eval_model_path='OpenSafetyLab/MD-Judge-v0.1'

GPU_NUMS=1

datestr=$(date +%Y%m%d_%H%M)      


for i in "${!models[@]}"; do
    path="${models[$i]}"
    model_name=$(basename "$path")

    outdir="results/${model_name}"
    
    echo "[Current path] $path"
    echo "[Current outdir] $outdir"
    start_time=$(date +%s)

    python gen_salad_bench.py --model_path=${path} --output_dir=${outdir} --tensor_parallel=${GPU_NUMS}
    python eval_salad_bench.py --model_path=${path} --output_dir=${outdir} --tensor_parallel=${GPU_NUMS} --eval_model_path ${eval_model_path}

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "${path},${dataset} time elapsed: $elapsed_time seconds"
    echo "------------------------------------------------------------"
    echo "------------------------------------------------------------"
done
