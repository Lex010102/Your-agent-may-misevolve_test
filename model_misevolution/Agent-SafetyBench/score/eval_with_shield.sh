# CUDA_VISIBLE_DEVICES=0 python eval_with_shield.py --model_path thu-coai/ShieldAgent --filepath ../evaluation/evaluation_results/tot-gpt4o --filename gen_res.json --label_type "" --batch_size 40 --target_model_name gpt4o


eval_model_path='thu-coai/ShieldAgent'


model_list=(
  llama3.1-70b-inst
  llama3.1-70b-inst-agentgen
)

for model_name in "${model_list[@]}"; do
  echo "Running evaluation for: $model_name"
  CUDA_VISIBLE_DEVICES=0 python eval_with_shield.py \
    --model_path $eval_model_path \
    --filepath ../evaluation/evaluation_results/tot-${model_name} \
    --filename gen_res.json \
    --label_type "" \
    --batch_size 40 \
    --target_model_name ${model_name}
done