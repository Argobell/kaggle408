MODEL="/root/autodl-tmp/kaggle408/checkpoints/gek_e2b" # Local path to model or huggingface id
DATA="/root/autodl-tmp/kaggle408/dataset/gek408_dpo" # Dataset path


OUTPUT_DIR="./checkpoints/dpo_ex1_$(date +%Y%m%d_%H%M%S)"

uv run python ./src/dpo_multi.py \
    --model_name_or_path "$MODEL" \
    --dataset_name "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    \
    --max_seq_length 2048 \
    --max_prompt_length 1024 \
    \
    --tune_vision \
    --tune_language_layers \
    --tune_attention_modules \
    --tune_mlp_modules \
    --lora_dropout 0.1 \
    --r 4 \
    --alpha 4 \
    \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    \
    --max_steps 100 \
    --learning_rate 5e-6 \
    --lr_scheduler_type "linear" \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --optim "adamw_torch_fused" \
    \
    --logging_steps 1 \
    --eval_steps 5 \
    --eval_strategy "steps" \
    --save_steps 25 \
    --save_strategy "steps" \
    --save_merged