MODEL="/root/autodl-tmp/model/gemma3n_E4B" # Local path to model or huggingface id
DATA="/root/autodl-tmp/kaggle408/dataset" # Dataset path

# It's assumed that the python script is located at ./src/sft_multi.py
# The user should replace "instruction" and "answer" with the actual column names in their dataset.
# The output directory is set to a timestamped folder to avoid overwriting results.
OUTPUT_DIR="./checkpoints/ex2_$(date +%Y%m%d_%H%M%S)"

uv run python ./src/sft_multi.py \
    --model_name_or_path "$MODEL" \
    --dataset_name "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    \
    --max_seq_length 2048 \
    \
    --tune_vision \
    --tune_language_layers \
    --tune_attention_modules \
    --tune_mlp_modules \
    --lora_dropout 0.05 \
    --r 16 \
    --alpha 16 \
    \
    --instruction_name "question" \
    --answer_name "solution" \
    \
    --do_eval \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    \
    --num_train_epochs 2 \
    --learning_rate 2e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 0.3 \
    --optim "adamw_torch_fused" \
    \
    --logging_steps 1 \
    --eval_steps 5 \
    --eval_strategy "steps" \
    --save_steps 20 \
    --save_strategy "steps" \
    --save_merged