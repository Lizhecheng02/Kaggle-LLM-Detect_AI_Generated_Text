python deberta_trainer_args.py \
    --version 1 \
    --is_load_from_disk False \
    --model_name "microsoft/deberta-v3-large" \
    --num_samples 1000000 \
    --test_size 0.10 \
    --random_state 2024 \
    --max_length 1024 \
    --num_train_epochs 2 \
    --learning_rate 4e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --fp16 True \
    --overwrite_output_dir True \
    --gradient_accumulation_steps 16 \
    --logging_steps 100 \
    --eval_steps 100 \
    --save_steps 100 \
    --lr_scheduler_type "linear" \
    --weight_decay 0.01 \
    --save_total_limit 3 \
    --num_warmup_steps 100 \
    --power 1.0 \
    --lr_end 2e-6