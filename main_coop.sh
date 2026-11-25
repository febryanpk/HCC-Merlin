#!/bin/bash

temperatures=0.07
tuning="coop"
lr=1e-1
n_ctx=(4)
n_shot=(1 2 4 8 16)

CHECKPOINT_DIR="path/to/checkpoint_dir"

# Loop n_ctx values
for ctx in "${n_shot[@]}"; do
    echo "========================================="
    echo "Running with n_ctx = ${n_ctx[0]}"
    echo "========================================="
    echo "========================================="
    echo "Training with $ctx shots"
    echo "========================================="
    SAVE_PATH="${CHECKPOINT_DIR}/Experiment1_${ctx}Shot_CSC_CoOp_${n_ctx[0]}CTX_lr${lr}_trainbs32_temp007frozen_epoch200_20260716.pth"

    python train.py \
        --phase "venous" \
        --epochs 200 \
        --train_batch_size 32 \
        --val_batch_size 50 \
        --tuning_mode "${tuning}" \
        --n_shot "$ctx" \
        --learning_rate "$lr" \
        --temperature 0.07 \
        --model_save_path "$SAVE_PATH" \
        --evaluate \
        --use_wandb \
        --csc \
        --n_ctx "${n_ctx[0]}"
done

# #!/bin/bash

# temperature=0.07
# tuning="coop"
# lr=2e-2
# n_ctx=(1 2 4 8 16)
# n_shot=(1 2 4 8 16)

# CHECKPOINT_DIR="path/to/checkpoint_dir"

# # Loop n_ctx values
# for ctx in "${n_ctx[@]}"; do
#     echo "========================================="
#     echo "Running with n_ctx = ${ctx}"
#     echo "========================================="

#     for shot in "${n_shot[@]}"; do
#         echo "========================================="
#         echo "Training with ${shot} shots"
#         echo "========================================="

#         SAVE_PATH="${CHECKPOINT_DIR}/${shot}Shot_CoOp_${ctx}CTX_lr${lr}_trainbs32_temp007frozen_epoch200_20260715.pth"

#         python train.py \
#             --phase "venous" \
#             --epochs 200 \
#             --train_batch_size 32 \
#             --val_batch_size 50 \
#             --tuning_mode "${tuning}" \
#             --n_shot "${shot}" \
#             --learning_rate "${lr}" \
#             --temperature "${temperature}" \
#             --model_save_path "${SAVE_PATH}" \
#             --evaluate \
#             --use_wandb \
#             --csc \
#             --n_ctx "${ctx}"

#     done
# done