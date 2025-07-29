MODEL="/root/autodl-tmp/kaggle408/checkpoints/gek_e2b" # Local path to model or huggingface id
DATA="/root/autodl-tmp/kaggle408/dataset/rlaif-v" # Dataset path


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
    --lora_dropout 0.05 \
    --r 16 \
    --alpha 16 \
    \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --optim "adamw_8bit" \
    \
    --logging_steps 5 \
    --eval_steps 20 \
    --eval_strategy "steps" \
    --save_steps 100 \
    --save_strategy "steps" \
    --save_merged