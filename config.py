import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class DiseasePrompts:
    disease_name: str
    positive_prompts: List[str]
    negative_prompts: List[str]


HCC_PROMPTS = DiseasePrompts(
    disease_name="hcc",
    positive_prompts=[
        "Liver mass consistent with hepatocellular carcinoma",
        "HCC detected with typical venous enhancement pattern",
        "Imaging shows signs of hepatocellular carcinoma",
    ],
    negative_prompts=[
        "No hepatocellular carcinoma found in liver",
        "No liver mass consistent with HCC detected",
        "Liver appears clear of hepatocellular carcinoma",
    ],
)


FRACTURE_PROMPTS = DiseasePrompts(
    disease_name="fracture",
    positive_prompts=[
        "compression fracture",
        "fracture identified",
        "fractures identified",
        "osteoporotic fracture",
        "vertebral fracture",
        "thoracic fracture",
        "thoracolumbar fracture",
        "wedge fractture",
        "biconcave fracture",
        "crush fracture",
    ],
    negative_prompts=[
        "no fracture",
        "no displaced fracture",
        "no acute fracture",
        "no evidence of fracture",
        "without evidence of fracture",
        "deformity and developmental abnormality",
    ],
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Merlin VLM with HCC and Negative CT data"
    )

    # Paths
    parser.add_argument(
        "--dataset_csv",
        type=str,
        default="path to dataset CSV",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="path to positive image root directory",
    )
    parser.add_argument(
        "--negative_root",
        type=str,
        default="path to negative image root directory",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="path to save trained model checkpoints",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--stop_epoch",
        type=int,
        default=None,
        help="Stop after this many epochs; LR schedule still spans --epochs",
    )
    parser.add_argument("--train_batch_size", type=int, default=18)
    parser.add_argument("--val_batch_size", type=int, default=18)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--phase",
        type=str,
        default="venous",
        choices=["venous", "all"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--lldr_mult",
        type=float,
        default=0.7,
        help="Layer-wise LR decay multiplier (1.0 = no decay)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=1,
        help="Epochs of linear LR warmup before cosine decay (0 = none)",
    )
    parser.add_argument("--text_lora_r", type=int, default=16)
    parser.add_argument("--text_lora_alpha", type=int, default=32)
    parser.add_argument("--image_lora_r", type=int, default=2)
    parser.add_argument("--image_lora_alpha", type=int, default=2)
    parser.add_argument(
        "--tuning_mode",
        type=str,
        default="full",
        # full: fine-tune both encoders; lora: LoRA on both encoders;
        # lora_text / lora_image / projection are the ablation settings
        choices=["full", "lora", "lora_text", "lora_image", "projection"],
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default="path to cache root directory",
        help="Directory holding interpolate_venous_phase_cache",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default=None,
        help="Where to write final test metrics",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stop after this many epochs without improvement",
    )
    parser.add_argument(
        "--early_stop_on",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_f1"],
        help="Metric whose improvement resets patience (val_f1 needs --evaluate)",
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip the final test evaluation and LoRA merge (hyperparameter search trials)",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="merlin_vlm")
    parser.add_argument("--wandb_group", type=str, default="Continual Learning")

    return parser.parse_args()
