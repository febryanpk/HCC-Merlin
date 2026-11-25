#!/bin/bash
# Ablation: which component of the LoRA setup matters?
#   lora_text  - LoRA on text encoder + text projection, image encoder frozen
#   lora_image - ConvLoRA on image encoder + image projection, text encoder frozen
#   projection - only the image and text projection layers trainable
#   lora       - full baseline: both LoRAs + both projections

temp=1.0
lr=1e-3
epochs=300     # LR schedule length (matches earlier runs)
stop_epoch=10  # ablations stop here; the full lora baseline uses early stopping
patience=5     # early stopping on val loss (full lora baseline only)

CACHE_ROOT="path/to/cache_root"
CHECKPOINT_DIR="path/to/checkpoint_dir/ablation"
mkdir -p "$CHECKPOINT_DIR"

# Override to re-run a subset, e.g. MODES=lora TAG=_v2 bash main_ablation.sh
MODES="${MODES:-lora_text lora_image projection lora}"
TAG="${TAG:-}"
EARLY_STOP_ON="${EARLY_STOP_ON:-val_loss}"  # metric the full lora baseline's patience watches

for mode in $MODES; do
    temp_tag=$(echo "$temp" | sed 's/\.//g')
    if [ "$mode" = "lora" ]; then
        stop_args=(--patience "$patience" --early_stop_on "$EARLY_STOP_ON")
        NAME="Ablation_${mode}_Venous_lr${lr}_bs18_temp${temp_tag}_epoch${epochs}_patience${patience}${TAG}"
    else
        stop_args=(--stop_epoch "$stop_epoch")
        NAME="Ablation_${mode}_Venous_lr${lr}_bs18_temp${temp_tag}_epoch${epochs}_stop${stop_epoch}${TAG}"
    fi

    echo "========================================="
    echo "Running ablation: $mode"
    echo "========================================="

    python train.py \
        --phase "venous" \
        --epochs "$epochs" \
        "${stop_args[@]}" \
        --train_batch_size 18 \
        --val_batch_size 18 \
        --tuning_mode "$mode" \
        --learning_rate "$lr" \
        --temperature "$temp" \
        --cache_root "$CACHE_ROOT" \
        --model_save_path "${CHECKPOINT_DIR}/${NAME}.pth" \
        --results_json "${CHECKPOINT_DIR}/${NAME}_results.json" \
        --evaluate \
        --use_wandb \
        --wandb_project "merlin_vlm" \
        --wandb_group "Ablation" \
        2>&1 | tee "${CHECKPOINT_DIR}/${NAME}.log"
done
