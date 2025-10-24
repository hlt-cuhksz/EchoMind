python -m torch.distributed.run --nproc_per_node=8 train.py \
    --deepspeed ds_config/dp_config_zero1.json \
    \
    --dataset_dirs "${data_dir}" \
    \
    --output_dir ${SAVE_ROOT} \
    --remove_unused_columns False \
    --seed 1 \
    --do_train True \
    --bf16 True \
    \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 1000 \
    \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    \
    --llm_model $llm_path \
    --audio_model $audio_path \
    --tts_model $tts_path \
    --unfreeze_adapter True \
    \
    --disable_tqdm True \
    --report_to "none" \
    \
    --logging_steps 20 \
    --save_steps 200 \
    --save_total_limit 1