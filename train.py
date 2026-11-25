import math
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch

import wandb
from config import FRACTURE_PROMPTS, HCC_PROMPTS, parse_args
from evaluation.engine import (
    classification_evaluation,
    evaluate_cocoop_model,
    evaluate_coop_model,
    evaluate_model,
    evaluate_recall_at_1,
)
from merlin import Merlin
from merlin.data.dataloaders import DataLoader, InterpolateDataLoader, VerseDataLoader
from merlin.data.preprocessing import (
    binarize_labels,
    build_verse_datalist,
    process_data,
    process_data_venous,
)
from merlin.models.lora_utils import (
    apply_image_lora,
    apply_text_lora,
    conv_merge_and_unload,
    freeze_module,
    unfreeze_module,
)
from merlin.utils.engine import train_one_epoch, validate_one_epoch
from merlin.utils.optimizer import (
    build_optimizer_and_scheduler,
    build_param_groups_image_text_lldr,
)
from utils import args

# warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Model Building


def is_coop(args):
    return args.tuning_mode == "coop"


def build_model(args, device):
    model = Merlin(use_coop=is_coop(args), n_ctx=args.n_ctx, csc=args.csc).to(device)

    if args.tuning_mode == "coop":
        # Text encoder + projection are frozen inside MerlinArchitecture;
        # only the prompt learner's context vectors are trained.
        freeze_module(model.model.encode_image.i3_resnet)
    elif args.tuning_mode == "lora":
        freeze_module(model)

        model.model.encode_text.text_encoder = apply_text_lora(
            model.model.encode_text.text_encoder,
            r=args.text_lora_r,
            lora_alpha=args.text_lora_alpha,
            lora_dropout=0.2,
        )

        model.model.encode_image.i3_resnet = apply_image_lora(
            model.model.encode_image.i3_resnet,
            r=args.image_lora_r,
            lora_alpha=args.image_lora_alpha,
        )
        unfreeze_module(model.model.encode_text.linear_layer)
        unfreeze_module(model.model.encode_image.i3_resnet.contrastive_head)
    # Ablations: adapt one tower (plus its projection) or only the projections
    elif args.tuning_mode == "lora_text":
        freeze_module(model)

        model.model.encode_text.text_encoder = apply_text_lora(
            model.model.encode_text.text_encoder,
            r=args.text_lora_r,
            lora_alpha=args.text_lora_alpha,
            lora_dropout=0.2,
        )
        unfreeze_module(model.model.encode_text.linear_layer)
    elif args.tuning_mode == "lora_image":
        freeze_module(model)

        model.model.encode_image.i3_resnet = apply_image_lora(
            model.model.encode_image.i3_resnet,
            r=args.image_lora_r,
            lora_alpha=args.image_lora_alpha,
        )
        unfreeze_module(model.model.encode_image.i3_resnet.contrastive_head)
    elif args.tuning_mode == "projection":
        freeze_module(model)

        unfreeze_module(model.model.encode_text.linear_layer)
        unfreeze_module(model.model.encode_image.i3_resnet.contrastive_head)

    model.model.temperature = torch.nn.Parameter(
        torch.tensor(args.temperature, device=device)
    )

    # Freeze the temperature parameter if using CoOp
    if is_coop(args):
        model.model.temperature.requires_grad = False

    _print_param_counts(model)
    return model


def _print_param_counts(model):
    img_params = sum(
        p.numel() for p in model.model.encode_image.parameters() if p.requires_grad
    )

    txt_params = sum(
        p.numel() for p in model.model.encode_text.parameters() if p.requires_grad
    )

    if hasattr(model.model, "prompt_learner"):
        coop_params = sum(
            p.numel()
            for p in model.model.prompt_learner.parameters()
            if p.requires_grad
        )
        txt_params += coop_params

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable image encoder params: {img_params:,}")
    print(f"Trainable text encoder params:  {txt_params:,}")  # Will now show 12,288
    print(f"Total trainable params:         {total:,}")


def _sample_fewshot_datalist(datalist, shots_per_class=16, seed=42):
    grouped = {}
    for item in datalist:
        grouped.setdefault(item["labels"], []).append(item)

    expected_labels = ["HCC", "negative"]
    missing = {
        label: len(grouped.get(label, []))
        for label in expected_labels
        if len(grouped.get(label, [])) < shots_per_class
    }
    if missing:
        raise ValueError(
            f"Not enough usable training samples for few-shot CoOp: {missing}. "
            "Check phase filtering and image paths."
        )

    rng = random.Random(seed)
    sampled = []
    for label in expected_labels:
        sampled.extend(rng.sample(grouped[label], shots_per_class))

    rng.shuffle(sampled)
    print(
        "Few-shot train samples: "
        + ", ".join(f"{label}={shots_per_class}" for label in expected_labels)
        + f" (total={len(sampled)})"
    )
    return sampled


# Data Loading
def build_dataloaders(args):
    df = pd.read_csv(args.dataset_csv)

    # Fill NaN with 'negative' for pathology and empty string for text fields
    df["patho"] = df["patho"].fillna("negative")

    df = df[df["patho"].isin(["HCC", "negative"])].reset_index(drop=True)

    if args.phase == "venous":
        train_datalist = process_data_venous(
            df[df["split"] == "train"], args.image_root, args.negative_root
        )
        val_datalist = process_data_venous(
            df[df["split"] == "val"],
            args.image_root,
            args.negative_root,
        )
        test_datalist = process_data_venous(
            df[df["split"] == "test"],
            args.image_root,
            args.negative_root,
        )
    else:
        train_datalist = process_data(
            df[df["split"] == "train"], args.image_root, args.negative_root
        )
        val_datalist = process_data(
            df[df["split"] == "val"],
            args.image_root,
            args.negative_root,
        )
        test_datalist = process_data(
            df[df["split"] == "test"],
            args.image_root,
            args.negative_root,
        )

    if is_coop(args):
        # Sample actual loadable training items after phase/path filtering.
        train_datalist = _sample_fewshot_datalist(
            train_datalist, shots_per_class=args.n_shot, seed=42
        )
        val_datalist = _sample_fewshot_datalist(
            val_datalist, shots_per_class=args.n_shot, seed=42
        )

    binarize_labels(train_datalist)
    binarize_labels(val_datalist)
    binarize_labels(test_datalist)

    if args.phase == "venous":
        cache_dir = os.path.join(args.cache_root, "interpolate_venous_phase_cache")
    else:
        cache_dir = None

    # Vertebral fracture loader (evaluation only)
    fracture_df = pd.read_excel("path to VerSe fracture annotation xlsx")
    fracture_df = fracture_df[fracture_df["dataset"] == "test secret"]
    verse_datalist = build_verse_datalist(
        fracture_df,
        fracture_dir="path to extracted VerSe CT directory",
    )

    train_loader = InterpolateDataLoader(
        train_datalist,
        batchsize=args.train_batch_size,
        cache_dir=cache_dir,
        shuffle=True,
        num_workers=0,
    )
    val_loader = InterpolateDataLoader(
        val_datalist,
        batchsize=args.train_batch_size,
        cache_dir=cache_dir,
        shuffle=False,
        num_workers=0,
    )
    test_loader = InterpolateDataLoader(
        test_datalist,
        batchsize=args.val_batch_size,
        cache_dir=cache_dir,
        shuffle=False,
        num_workers=0,
    )
    verse_loader = VerseDataLoader(
        datalist=verse_datalist,
        cache_dir=None,
        batchsize=1,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, verse_loader, test_loader


### Training


def train(args, epoch_callback=None):
    """Train and return the results dict (non-CoOp).

    epoch_callback(epoch, epoch_log, best_val_f1) runs after every epoch; the
    hyperparameter search uses it to report to Optuna and prune trials.
    """
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity="your wandb entity",
            group=args.wandb_group,
            name=os.path.splitext(os.path.basename(args.model_save_path))[0],
            # tags=["continual_learning", "i3_resnet", "clinical_longformer"],
            config=vars(args),
        )

    model = build_model(args, device)
    train_loader, val_loader, verse_loader, test_loader = build_dataloaders(args)

    # Build LLDR param groups
    image_layer_names = [n for n, _ in model.model.encode_image.named_parameters()][
        ::-1
    ]
    text_layer_names = [n for n, _ in model.model.encode_text.named_parameters()][::-1]

    param_groups = build_param_groups_image_text_lldr(
        model=model,
        image_layer_names=image_layer_names,
        text_layer_names=text_layer_names,
        base_lr_img=args.learning_rate,
        base_lr_txt=args.learning_rate,
        lr_mult_img=args.lldr_mult,
        lr_mult_txt=args.lldr_mult,
        weight_decay=args.weight_decay,
    )
    accumulation_steps = 1 if is_coop(args) else 8
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = args.epochs * steps_per_epoch

    # Print trainable parameter in optimizer groups for verification
    # total_opt_params = 0
    # for group in param_groups:
    #     group_params = sum(p.numel() for p in group["params"])
    #     total_opt_params += group_params
    #     print(f"Param group '{group['name']}': {group_params:,} params")

    # 1% warmup
    # warmup_steps = int(0.1 * total_steps)

    # Warmup for --warmup_epochs epochs of optimizer updates
    warmup_steps = args.warmup_epochs * steps_per_epoch

    if is_coop(args):
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9
        )
        # Scheduler: Cosine Annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )
        scaler = torch.cuda.amp.GradScaler()
    else:
        optimizer, scheduler, scaler = build_optimizer_and_scheduler(
            model, param_groups, args, total_steps, warmup_steps=warmup_steps
        )
        # classification_loss_fn = torch.nn.BCEWithLogitsLoss()

    best_f1 = best_acc = 0.0
    best_val_loss = float("inf")

    best_loss_path = args.model_save_path.replace(".pth", "_best_loss.pth")
    best_f1_path = args.model_save_path.replace(".pth", "_best_f1.pth")
    best_acc_path = args.model_save_path.replace(".pth", "_best_acc.pth")

    best_loss_epoch = best_f1_epoch = None
    patience_counter = 0
    global_step = 0

    n_epochs = min(args.epochs, args.stop_epoch or args.epochs)
    for epoch in range(n_epochs):
        (
            train_loss,
            global_step,
            # cls_loss,
            ctr_loss,
            train_img_feats,
            train_txt_feats,
            train_cos,
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            # classification_loss_fn,
            epoch,
            device,
            global_step,
            scheduler,
            args,
            accumulation_steps=accumulation_steps,
        )

        (
            val_loss,
            global_step,
            # val_cls_loss,
            val_ctr_loss,
            val_img_feats,
            val_txt_feats,
            val_cos,
        ) = validate_one_epoch(
            model,
            val_loader,
            # classification_loss_fn,
            epoch,
            device,
            global_step,
            args,
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        # if is_coop(args):
        #     (
        #         train_f1_score,
        #         train_lower,
        #         train_upper,
        #         train_recall,
        #         train_precision,
        #         train_accuracy,
        #         _,
        #         _,
        #     ) = evaluate_coop_model(model, train_loader, device)
        #     (
        #         val_f1_score,
        #         val_lower,
        #         val_upper,
        #         val_recall,
        #         val_precision,
        #         val_accuracy,
        #         _,
        #         _,
        #     ) = evaluate_coop_model(model, val_loader, device)
        #     print(f"Train F1: {train_f1_score:.4f} | Val F1: {val_f1_score:.4f}")
        #     print(
        #         f"Train Accuracy: {train_accuracy:.4f} | Val Accuracy: {val_accuracy:.4f}"
        #     )

        epoch_log = {
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "temperature": model.model.temperature.item(),
            "epoch": epoch,
        }

        # Optional per-epoch evaluation
        f1_score = 0.0
        if args.evaluate:
            # acc_cls, rec_cls, prec_cls, f1_cls = classification_evaluation(
            # model, val_loader, device
            # )

            # Zero-shot prompt evaluation (CoOp is evaluated on its learned prompts)
            if not is_coop(args):
                f1_score, lower, upper, recall, precision, accuracy, _, _, auc = (
                    evaluate_model(model, val_loader, device, HCC_PROMPTS)
                )
                f1_frac, lo_frac, hi_frac, rec_frac, prec_frac, acc_frac, _, _, _ = (
                    evaluate_model(model, verse_loader, device, FRACTURE_PROMPTS)
                )
                (train_recall_i2t, train_recall_t2i) = evaluate_recall_at_1(
                    torch.tensor(train_img_feats),
                    torch.tensor(train_txt_feats),
                    pool_size=64,
                    allow_partial_pool=True,
                )

                (val_recall_i2t, val_recall_t2i) = evaluate_recall_at_1(
                    torch.tensor(val_img_feats),
                    torch.tensor(val_txt_feats),
                    pool_size=64,
                    allow_partial_pool=True,
                )
                epoch_log.update(
                    {
                        "evaluate/F1-Score": f1_score,
                        "evaluate/F1-Lower": lower,
                        "evaluate/F1-Upper": upper,
                        "evaluate/Recall": recall,
                        "evaluate/Precision": precision,
                        "evaluate/Accuracy": accuracy,
                        "evaluate/AUC": auc,
                        "evaluate_fracture/F1-Score": f1_frac,
                        "val/Recall@1-ImageToText": val_recall_i2t,
                        "val/Recall@1-TextToImage": val_recall_t2i,
                        "train/Recall@1-ImageToText": train_recall_i2t,
                        "train/Recall@1-TextToImage": train_recall_t2i,
                    }
                )

        if args.use_wandb:
            wandb.log(epoch_log)

        # Keep both the best val loss and the best zero-shot val F1 checkpoints
        loss_improved = val_loss < best_val_loss
        if loss_improved:
            best_val_loss = val_loss
            best_loss_epoch = epoch + 1
            torch.save(model.state_dict(), best_loss_path)
            print(f"Saved best model (Val Loss={val_loss:.4f})")
        f1_improved = args.evaluate and not is_coop(args) and f1_score > best_f1
        if f1_improved:
            best_f1 = f1_score
            best_f1_epoch = epoch + 1
            torch.save(model.state_dict(), best_f1_path)
            print(f"Saved best F1 model (Val F1={f1_score:.4f})")

        # Early stopping watches the metric chosen by --early_stop_on
        if f1_improved if args.early_stop_on == "val_f1" else loss_improved:
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No {args.early_stop_on} improvement (Best Val Loss={best_val_loss:.4f}, "
                f"Best Val F1={best_f1:.4f}, patience {patience_counter}/{args.patience})"
            )

        if epoch_callback is not None:
            epoch_callback(epoch, epoch_log, best_f1)

        # if is_coop(args):
        #     # Checkpoint on best accuracy
        #     if accuracy > best_acc:
        #         best_acc = accuracy
        #         torch.save(model.state_dict(), best_acc_path)
        #         print(f"Saved best model (Accuracy={accuracy:.4f})")
        #         patience_counter = 0
        #     else:
        #         print(f"No improvement (best Accuracy={best_acc:.4f})")
        #         patience_counter += 1

        # Early stopping
        if args.patience and patience_counter >= args.patience:
            print("Early stopping triggered.")
            break
    if is_coop(args):
        # f1_score, lower, upper, recall, precision, accuracy, _, _ = evaluate_coop_model(
        # model, test_loader, device
        # )
        # Evaluate using best loss model
        model.load_state_dict(torch.load(best_loss_path))
        f1_score, lower, upper, recall, precision, accuracy, _, _ = evaluate_coop_model(
            model, test_loader, device
        )
        # f1_score, lower, upper, recall, precision, accuracy, _, _ = (
        # evaluate_cocoop_model(model, val_loader, device)
        # )
        if args.use_wandb:
            wandb.log(
                {
                    "evaluate/F1-Score": f1_score,
                    "evaluate/F1-Lower": lower,
                    "evaluate/F1-Upper": upper,
                    "evaluate/Recall": recall,
                    "evaluate/Precision": precision,
                    "evaluate/Accuracy": accuracy,
                }
            )

    else:
        results = {
            "tuning_mode": args.tuning_mode,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "lldr_mult": args.lldr_mult,
            "warmup_epochs": args.warmup_epochs,
            "text_lora_r": args.text_lora_r,
            "text_lora_alpha": args.text_lora_alpha,
            "image_lora_r": args.image_lora_r,
            "image_lora_alpha": args.image_lora_alpha,
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "best_val_loss": best_val_loss,
            "best_loss_epoch": best_loss_epoch,
            "best_val_f1": best_f1,
            "best_f1_epoch": best_f1_epoch,
        }
        # Zero-shot HCC prompts on the held-out test split, for each checkpoint
        checkpoints = {"best_loss": best_loss_path}
        if best_f1_epoch is not None:
            checkpoints["best_f1"] = best_f1_path
        if args.skip_test:
            checkpoints = {}
        for tag, path in checkpoints.items():
            model.load_state_dict(torch.load(path))
            f1_score, lower, upper, recall, precision, accuracy, _, _, auc = (
                evaluate_model(model, test_loader, device, HCC_PROMPTS)
            )
            results[f"test_{tag}"] = {
                "f1": f1_score,
                "f1_lower": lower,
                "f1_upper": upper,
                "recall": recall,
                "precision": precision,
                "accuracy": accuracy,
                "auc": auc,
            }
            if args.use_wandb:
                wandb.log(
                    {f"test_{tag}/{k}": v for k, v in results[f"test_{tag}"].items()}
                )
        print(results)
        if args.results_json:
            import json

            with open(args.results_json, "w") as f:
                json.dump(results, f, indent=2)

    # Merge LoRA adapters and save clean model
    text_lora = args.tuning_mode in ("lora", "lora_text")
    image_lora = args.tuning_mode in ("lora", "lora_image")

    def _merge(path):
        model = build_model(args, device)
        model.load_state_dict(torch.load(path))
        if text_lora:
            model.model.encode_text.text_encoder = (
                model.model.encode_text.text_encoder.merge_and_unload()
            )
        if image_lora:
            model.model.encode_image.i3_resnet = conv_merge_and_unload(
                model.model.encode_image.i3_resnet
            )
        merged_path = path.replace(".pth", "_merged.pth")
        torch.save(model.state_dict(), merged_path)
        print(f"Merged model saved to {merged_path}")

    if (text_lora or image_lora) and not args.skip_test:
        _merge(best_loss_path)
        if best_f1_epoch is not None:
            _merge(best_f1_path)

    if args.use_wandb:
        wandb.finish()

    if not is_coop(args):
        return results


if __name__ == "__main__":
    train(parse_args())
