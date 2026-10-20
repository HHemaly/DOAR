"""GENERATED FILE -- do not edit by hand.

Verbatim function bodies extracted by experiments/journal/tools/extract_frozen_e2.py
from DOAR_21_8_26_V1.ipynb (sha256 b7edd22e84ccef6554018ce6883dff20d2445459bd813083dd2d6686dd5f4ac4), cell 6 (+ evaluate from cell 7) (E1-B controlled representation adaptation).
Thresholds, constants and logic are unchanged from the runs reported in the
archived notebook; only module imports were added so the functions can run
outside Colab.
"""
# flake8: noqa
# ruff: noqa
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import csv
import gc
import json
import random
import time

import pandas as pd
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageOps
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    import open_clip
except ImportError:  # CPU test environments without open_clip
    open_clip = None

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            h.update(chunk)

    return h.hexdigest()

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def conservative_augmentation():

    return [
        transforms.RandomAffine(
            degrees=7,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
        ),

        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.05,
        ),
    ]

class DrawingDataset(Dataset):

    def __init__(
        self,
        rows,
        transform
    ):

        self.rows = rows
        self.transform = transform

    def __len__(self):

        return len(
            self.rows
        )

    def __getitem__(
        self,
        index
    ):

        row = self.rows[index]

        with Image.open(
            row["path"]
        ) as image_file:

            image = image_file.convert(
                "RGB"
            )

        image = self.transform(
            image
        )

        target = CLASS_TO_INDEX[
            row["class"]
        ]

        return (
            image,
            target,
            row["image_id"],
        )

def build_efficientnet_transforms():

    weights = (
        models
        .EfficientNet_B0_Weights
        .DEFAULT
    )

    native = weights.transforms()

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                native.resize_size,
                interpolation=
                    native.interpolation,
            ),

            transforms.CenterCrop(
                native.crop_size
            ),

            *conservative_augmentation(),

            transforms.ToTensor(),

            transforms.Normalize(
                native.mean,
                native.std,
            ),
        ]
    )

    return (
        train_transform,
        native,
        weights,
    )

def build_openclip_train_transform(
    native_eval
):

    assert hasattr(
        native_eval,
        "transforms"
    )

    original_steps = list(
        native_eval.transforms
    )

    output_steps = []

    inserted = False

    tensor_step_names = {
        "ToTensor",
        "PILToTensor",
        "ToImage",
    }

    for step in original_steps:

        name = (
            step.__class__.__name__
        )

        if (
            not inserted
            and name in tensor_step_names
        ):

            output_steps.extend(
                conservative_augmentation()
            )

            inserted = True

        output_steps.append(step)

    if not inserted:

        output_steps = []

        for step in original_steps:

            name = (
                step.__class__.__name__
            )

            if (
                not inserted
                and name == "Normalize"
            ):

                output_steps.extend(
                    conservative_augmentation()
                )

                inserted = True

            output_steps.append(step)

    if not inserted:

        raise RuntimeError(
            "Could not insert augmentation "
            "into OpenCLIP preprocessing."
        )

    return transforms.Compose(
        output_steps
    )

class OpenCLIPClassifier(nn.Module):

    def __init__(
        self,
        clip_model,
        embedding_dimension
    ):

        super().__init__()

        self.clip_model = clip_model

        self.head = nn.Linear(
            embedding_dimension,
            len(CLASSES)
        )

    @property
    def visual(self):

        return self.clip_model.visual

    def forward(self, images):

        features = (
            self.clip_model
            .encode_image(
                images,
                normalize=False
            )
            .flatten(1)
        )

        return self.head(
            features
        )

def infer_openclip_embedding_dimension(
    clip_model,
    eval_transform,
    sample_path
):

    with Image.open(
        sample_path
    ) as image_file:

        image = image_file.convert(
            "RGB"
        )

    tensor = (
        eval_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    clip_model = clip_model.to(
        DEVICE
    )

    clip_model.eval()

    with torch.no_grad():

        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE
        ):

            features = (
                clip_model
                .encode_image(
                    tensor,
                    normalize=False
                )
                .flatten(1)
            )

    dimension = int(
        features.shape[1]
    )

    del tensor
    del features

    torch.cuda.empty_cache()

    return dimension

def freeze_module(module):

    for parameter in (
        module.parameters()
    ):

        parameter.requires_grad = False

def unfreeze_module(module):

    for parameter in (
        module.parameters()
    ):

        parameter.requires_grad = True

def parameter_count(module):

    return sum(
        p.numel()
        for p in module.parameters()
    )

def trainable_parameter_count(
    module
):

    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )

def ordered_visual_units(
    visual
):

    if (
        hasattr(
            visual,
            "transformer"
        )
        and hasattr(
            visual.transformer,
            "resblocks"
        )
    ):

        return list(
            visual
            .transformer
            .resblocks
        )

    if (
        hasattr(
            visual,
            "transformer"
        )
        and hasattr(
            visual.transformer,
            "blocks"
        )
    ):

        return list(
            visual
            .transformer
            .blocks
        )

    if (
        hasattr(
            visual,
            "trunk"
        )
        and hasattr(
            visual.trunk,
            "blocks"
        )
    ):

        return list(
            visual.trunk.blocks
        )

    if hasattr(
        visual,
        "blocks"
    ):

        return list(
            visual.blocks
        )

    if (
        hasattr(
            visual,
            "trunk"
        )
        and hasattr(
            visual.trunk,
            "stages"
        )
    ):

        return list(
            visual.trunk.stages
        )

    if hasattr(
        visual,
        "stages"
    ):

        return list(
            visual.stages
        )

    return None

def configure_openclip_trainability(
    wrapper,
    mode
):

    freeze_module(
        wrapper.clip_model
    )

    unfreeze_module(
        wrapper.head
    )

    visual = wrapper.visual

    if mode == "frozen":

        return

    if mode == "full":

        unfreeze_module(
            visual
        )

        return

    if mode != "partial":

        raise ValueError(mode)

    units = ordered_visual_units(
        visual
    )

    if not units:

        raise RuntimeError(
            "Cannot identify visual blocks."
        )

    visual_total = parameter_count(
        visual
    )

    target = (
        visual_total *
        PARTIAL_TARGET
    )

    accumulated = 0

    selected = []

    for unit in reversed(
        units
    ):

        selected.append(
            unit
        )

        accumulated += (
            parameter_count(unit)
        )

        if accumulated >= target:
            break

    for unit in selected:

        unfreeze_module(
            unit
        )

    for attribute_name in [
        "ln_post",
        "norm",
        "fc_norm",
        "head",
    ]:

        item = getattr(
            visual,
            attribute_name,
            None
        )

        if isinstance(
            item,
            nn.Module
        ):

            unfreeze_module(
                item
            )

    projection = getattr(
        visual,
        "proj",
        None
    )

    if isinstance(
        projection,
        nn.Parameter
    ):

        projection.requires_grad = True

def configure_efficientnet_trainability(
    model,
    mode
):

    freeze_module(
        model
    )

    unfreeze_module(
        model.classifier
    )

    if mode == "frozen":

        return

    if mode == "full":

        unfreeze_module(
            model.features
        )

        return

    if mode != "partial":

        raise ValueError(mode)

    stages = list(
        model.features
    )

    visual_total = sum(
        parameter_count(stage)
        for stage in stages
    )

    target = (
        visual_total *
        PARTIAL_TARGET
    )

    accumulated = 0

    selected = []

    for stage in reversed(
        stages
    ):

        selected.append(
            stage
        )

        accumulated += (
            parameter_count(stage)
        )

        if accumulated >= target:
            break

    for stage in selected:

        unfreeze_module(
            stage
        )

def build_model_and_transforms(
    model_key,
    mode
):

    if model_key == "efficientnet_b0":

        (
            train_transform,
            eval_transform,
            weights
        ) = (
            build_efficientnet_transforms()
        )

        model = (
            models.efficientnet_b0(
                weights=weights
            )
        )

        embedding_dimension = (
            model
            .classifier[-1]
            .in_features
        )

        model.classifier[-1] = (
            nn.Linear(
                embedding_dimension,
                len(CLASSES)
            )
        )

        configure_efficientnet_trainability(
            model,
            mode
        )

        return (
            model,
            train_transform,
            eval_transform,
            embedding_dimension
        )

    if model_key == "clip_vit_b32":

        openclip_name = (
            "ViT-B-32-quickgelu"
        )

        pretrained = "openai"

    elif model_key == "siglip2_vit_b16":

        openclip_name = (
            "ViT-B-16-SigLIP2"
        )

        pretrained = "webli"

    else:

        raise ValueError(
            model_key
        )

    (
        clip_model,
        _native_train,
        native_eval
    ) = (
        open_clip
        .create_model_and_transforms(
            openclip_name,
            pretrained=pretrained
        )
    )

    train_transform = (
        build_openclip_train_transform(
            native_eval
        )
    )

    embedding_dimension = (
        infer_openclip_embedding_dimension(
            clip_model,
            native_eval,
            train_rows[0]["path"]
        )
    )

    wrapper = OpenCLIPClassifier(
        clip_model,
        embedding_dimension
    )

    configure_openclip_trainability(
        wrapper,
        mode
    )

    return (
        wrapper,
        train_transform,
        native_eval,
        embedding_dimension
    )

def set_training_mode_respecting_freeze(
    model
):

    model.train()

    modules = list(
        model.modules()
    )

    # Larger parents first.
    for module in modules[1:]:

        parameters = list(
            module.parameters(
                recurse=True
            )
        )

        if (
            parameters
            and not any(
                p.requires_grad
                for p in parameters
            )
        ):

            module.eval()

def parameter_report(
    model,
    model_key
):

    if model_key == "efficientnet_b0":

        visual = model.features
        head = model.classifier

    else:

        visual = model.visual
        head = model.head

    visual_total = (
        parameter_count(
            visual
        )
    )

    visual_trainable = (
        trainable_parameter_count(
            visual
        )
    )

    return {
        "total_model_params":
            int(
                parameter_count(model)
            ),

        "total_trainable_params":
            int(
                trainable_parameter_count(
                    model
                )
            ),

        "visual_encoder_params":
            int(
                visual_total
            ),

        "visual_trainable_params":
            int(
                visual_trainable
            ),

        "visual_trainable_percent":
            float(
                100.0 *
                visual_trainable /
                max(
                    1,
                    visual_total
                )
            ),

        "head_params":
            int(
                parameter_count(head)
            ),
    }

def build_optimizer(
    model,
    model_key
):

    head_parameters = []

    encoder_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if model_key == "efficientnet_b0":

            is_head = (
                name.startswith(
                    "classifier."
                )
            )

        else:

            is_head = (
                name.startswith(
                    "head."
                )
            )

        if is_head:

            head_parameters.append(
                parameter
            )

        else:

            encoder_parameters.append(
                parameter
            )

    groups = []

    if head_parameters:

        groups.append(
            {
                "params":
                    head_parameters,

                "lr":
                    HEAD_LR,
            }
        )

    if encoder_parameters:

        groups.append(
            {
                "params":
                    encoder_parameters,

                "lr":
                    ENCODER_LR,
            }
        )

    return torch.optim.AdamW(
        groups,
        weight_decay=
            WEIGHT_DECAY
    )

def compute_class_weights(
    rows
):

    targets = np.asarray(
        [
            CLASS_TO_INDEX[
                row["class"]
            ]
            for row in rows
        ]
    )

    counts = np.bincount(
        targets,
        minlength=
            len(CLASSES)
    ).astype(
        np.float64
    )

    return (
        counts.sum() /
        (
            len(CLASSES) *
            np.clip(
                counts,
                1,
                None
            )
        )
    )

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_f1,
    stale,
    history,
    metadata
):

    torch.save(
        {
            "model_state":
                model.state_dict(),

            "optimizer_state":
                optimizer.state_dict(),

            "epoch":
                epoch,

            "best_valid_macro_f1":
                best_f1,

            "stale_epochs":
                stale,

            "history":
                history,

            "metadata":
                metadata,
        },
        path
    )

def run_one(
    model_key,
    mode,
    seed
):

    run_name = (
        f"{model_key}__"
        f"{mode}__"
        f"seed_{seed}"
    )

    run_dir = (
        OUT_ROOT /
        "runs" /
        run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    result_path = (
        run_dir /
        "result.json"
    )

    # ----------------------------------------------------------------------------------------------
    # RESUME — completed result
    # ----------------------------------------------------------------------------------------------

    if result_path.exists():

        result = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            result.get("status")
            == "complete"
        ):

            print(
                f"✓ SKIP completed: "
                f"{run_name}"
            )

            return result

    print()
    print("=" * 110)
    print(
        f"RUN: {run_name}"
    )
    print("=" * 110)

    set_seed(
        seed
    )

    gc.collect()

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    (
        model,
        train_transform,
        eval_transform,
        embedding_dimension
    ) = (
        build_model_and_transforms(
            model_key,
            mode
        )
    )

    model = model.to(
        DEVICE
    )

    params = parameter_report(
        model,
        model_key
    )

    print(
        "Embedding dimension:",
        embedding_dimension
    )

    print(
        "Visual trainable:",
        f"{params['visual_trainable_percent']:.2f}%"
    )

    # ----------------------------------------------------------------------------------------------
    # Scientific protocol assertions
    # ----------------------------------------------------------------------------------------------

    if mode == "frozen":

        assert (
            params[
                "visual_trainable_params"
            ]
            == 0
        )

    elif mode == "partial":

        assert (
            0
            <
            params[
                "visual_trainable_percent"
            ]
            <
            100
        )

    elif mode == "full":

        assert (
            abs(
                params[
                    "visual_trainable_percent"
                ]
                - 100
            )
            < 1e-6
        )

    train_dataset = (
        DrawingDataset(
            train_rows,
            train_transform
        )
    )

    valid_dataset = (
        DrawingDataset(
            valid_rows,
            eval_transform
        )
    )

    generator = (
        torch.Generator()
        .manual_seed(
            seed
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=
            BATCH_SIZE,
        shuffle=True,
        generator=
            generator,
        num_workers=
            NUM_WORKERS,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=
            BATCH_SIZE,
        shuffle=False,
        num_workers=
            NUM_WORKERS,
        pin_memory=True
    )

    class_weights = torch.tensor(
        CLASS_WEIGHTS,
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=
            class_weights,

        label_smoothing=
            LABEL_SMOOTHING
    )

    optimizer = build_optimizer(
        model,
        model_key
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            AMP_DTYPE
            == torch.float16
        )
    )

    last_path = (
        run_dir /
        "last.pt"
    )

    best_path = (
        run_dir /
        "best.pt"
    )

    history = []

    best_f1 = -1.0
    best_epoch = -1
    stale = 0
    start_epoch = 0

    # ----------------------------------------------------------------------------------------------
    # Resume interrupted training
    # ----------------------------------------------------------------------------------------------

    if last_path.exists():

        payload = torch.load(
            last_path,
            map_location=DEVICE,
            weights_only=False
        )

        metadata = payload.get(
            "metadata",
            {}
        )

        if (
            metadata.get("model")
            == model_key
            and metadata.get("mode")
            == mode
            and metadata.get("seed")
            == seed
        ):

            model.load_state_dict(
                payload[
                    "model_state"
                ]
            )

            optimizer.load_state_dict(
                payload[
                    "optimizer_state"
                ]
            )

            start_epoch = (
                payload["epoch"] + 1
            )

            best_f1 = payload[
                "best_valid_macro_f1"
            ]

            stale = payload[
                "stale_epochs"
            ]

            history = payload[
                "history"
            ]

            if history:

                best_epoch = max(
                    history,
                    key=lambda x:
                        x["val_macro_f1"]
                )["epoch"]

            print(
                "Resuming at epoch",
                start_epoch
            )

    training_started = (
        time.perf_counter()
    )

    # ----------------------------------------------------------------------------------------------
    # Epoch loop
    # ----------------------------------------------------------------------------------------------

    for epoch in range(
        start_epoch,
        MAX_EPOCHS
    ):

        epoch_started = (
            time.perf_counter()
        )

        set_training_mode_respecting_freeze(
            model
        )

        train_loss = 0.0
        train_correct = 0
        train_n = 0

        for (
            images,
            labels,
            _ids
        ) in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=AMP_DTYPE
            ):

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    labels
                )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite training loss."
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for p
                    in model.parameters()
                    if p.requires_grad
                ],
                max_norm=1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            train_loss += (
                float(
                    loss.detach()
                ) *
                len(labels)
            )

            train_correct += int(
                (
                    logits.argmax(1)
                    == labels
                ).sum()
            )

            train_n += (
                len(labels)
            )

        validation = evaluate(
            model,
            valid_loader,
            criterion
        )

        val_macro_f1 = (
            f1_score(
                validation["truth"],
                validation["prediction"],
                average="macro",
                zero_division=0
            )
        )

        val_accuracy = (
            accuracy_score(
                validation["truth"],
                validation["prediction"]
            )
        )

        val_balanced = (
            balanced_accuracy_score(
                validation["truth"],
                validation["prediction"]
            )
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_started
        )

        is_best = (
            val_macro_f1
            > best_f1
        )

        record = {
            "model":
                model_key,

            "mode":
                mode,

            "seed":
                seed,

            "epoch":
                epoch,

            "train_loss":
                train_loss /
                max(
                    1,
                    train_n
                ),

            "train_accuracy":
                train_correct /
                max(
                    1,
                    train_n
                ),

            "val_loss":
                validation[
                    "loss"
                ],

            "val_accuracy":
                val_accuracy,

            "val_balanced_accuracy":
                val_balanced,

            "val_macro_f1":
                val_macro_f1,

            "runtime_seconds":
                epoch_seconds,

            "is_best_epoch":
                is_best,
        }

        history.append(
            record
        )

        if is_best:

            best_f1 = (
                val_macro_f1
            )

            best_epoch = (
                epoch
            )

            stale = 0

            torch.save(
                {
                    "model_state":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "model":
                        model_key,

                    "mode":
                        mode,

                    "seed":
                        seed,

                    "embedding_dimension":
                        embedding_dimension,

                    "val_macro_f1":
                        val_macro_f1,

                    "parameter_report":
                        params,
                },
                best_path
            )

            prediction_df = pd.DataFrame(
                {
                    "image_id":
                        validation[
                            "image_ids"
                        ],

                    "true_class":
                        [
                            CLASSES[i]
                            for i
                            in validation[
                                "truth"
                            ]
                        ],

                    "predicted_class":
                        [
                            CLASSES[i]
                            for i
                            in validation[
                                "prediction"
                            ]
                        ],
                }
            )

            for i, class_name in enumerate(
                CLASSES
            ):

                prediction_df[
                    f"prob_{class_name}"
                ] = (
                    validation[
                        "probabilities"
                    ][:, i]
                )

            prediction_df.to_csv(
                run_dir /
                "best_valid_predictions.csv",
                index=False
            )

        else:

            stale += 1

        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            best_f1,
            stale,
            history,
            {
                "model":
                    model_key,

                "mode":
                    mode,

                "seed":
                    seed,

                "source_commit":
                    SOURCE_COMMIT,

                "t0_sha":
                    EXPECTED_T0_SHA,
            }
        )

        print(
            f"epoch={epoch + 1:02d}/"
            f"{MAX_EPOCHS} | "
            f"train_loss="
            f"{record['train_loss']:.4f} | "
            f"train_acc="
            f"{record['train_accuracy']:.4f} | "
            f"val_loss="
            f"{record['val_loss']:.4f} | "
            f"val_acc="
            f"{val_accuracy:.4f} | "
            f"val_macro_f1="
            f"{val_macro_f1:.4f}"
            + (
                " ★ BEST"
                if is_best
                else ""
            )
        )

        if stale >= PATIENCE:

            print(
                "Early stopping."
            )

            break

    training_seconds = (
        time.perf_counter()
        - training_started
    )

    peak_vram_mb = (
        torch.cuda
        .max_memory_allocated()
        / 1e6
    )

    # ----------------------------------------------------------------------------------------------
    # Reload best checkpoint
    # ----------------------------------------------------------------------------------------------

    best_payload = torch.load(
        best_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        best_payload[
            "model_state"
        ]
    )

    final_validation = evaluate(
        model,
        valid_loader,
        criterion
    )

    final_macro_f1 = f1_score(
        final_validation["truth"],
        final_validation["prediction"],
        average="macro",
        zero_division=0
    )

    assert abs(
        final_macro_f1
        - best_f1
    ) < 1e-8

    (
        precision,
        recall,
        class_f1,
        support
    ) = (
        precision_recall_fscore_support(
            final_validation["truth"],
            final_validation["prediction"],
            labels=
                list(
                    range(
                        len(CLASSES)
                    )
                ),
            zero_division=0
        )
    )

    # ----------------------------------------------------------------------------------------------
    # Per-class table
    # ----------------------------------------------------------------------------------------------

    class_rows = []

    for class_index, class_name in enumerate(
        CLASSES
    ):

        class_rows.append(
            {
                "model":
                    model_key,

                "mode":
                    mode,

                "seed":
                    seed,

                "class":
                    class_name,

                "precision":
                    float(
                        precision[
                            class_index
                        ]
                    ),

                "recall":
                    float(
                        recall[
                            class_index
                        ]
                    ),

                "f1":
                    float(
                        class_f1[
                            class_index
                        ]
                    ),

                "support":
                    int(
                        support[
                            class_index
                        ]
                    ),
            }
        )

    pd.DataFrame(
        class_rows
    ).to_csv(
        run_dir /
        "per_class_metrics.csv",
        index=False
    )

    # ----------------------------------------------------------------------------------------------
    # Confusion matrix
    # ----------------------------------------------------------------------------------------------

    cm = confusion_matrix(
        final_validation[
            "truth"
        ],
        final_validation[
            "prediction"
        ],
        labels=
            list(
                range(
                    len(CLASSES)
                )
            )
    )

    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES
    ).to_csv(
        run_dir /
        "confusion_matrix.csv"
    )

    # ----------------------------------------------------------------------------------------------
    # History
    # ----------------------------------------------------------------------------------------------

    pd.DataFrame(
        history
    ).to_csv(
        run_dir /
        "epoch_history.csv",
        index=False
    )

    # ----------------------------------------------------------------------------------------------
    # Qualitative fixed-example predictions
    # ----------------------------------------------------------------------------------------------

    best_predictions = pd.read_csv(
        run_dir /
        "best_valid_predictions.csv"
    )

    qualitative_predictions = (
        best_predictions[
            best_predictions[
                "image_id"
            ]
            .astype(str)
            .isin(
                QUAL_IDS
            )
        ]
        .copy()
    )

    qualitative_predictions[
        "model"
    ] = model_key

    qualitative_predictions[
        "mode"
    ] = mode

    qualitative_predictions[
        "seed"
    ] = seed

    qualitative_predictions.to_csv(
        run_dir /
        "qualitative_predictions.csv",
        index=False
    )

    result = {
        "status":
            "complete",

        "model":
            model_key,

        "mode":
            mode,

        "seed":
            seed,

        "best_epoch":
            int(
                best_epoch
            ),

        "epochs_completed":
            len(history),

        "best_valid_macro_f1":
            float(
                best_f1
            ),

        "val_accuracy":
            float(
                accuracy_score(
                    final_validation[
                        "truth"
                    ],
                    final_validation[
                        "prediction"
                    ]
                )
            ),

        "val_balanced_accuracy":
            float(
                balanced_accuracy_score(
                    final_validation[
                        "truth"
                    ],
                    final_validation[
                        "prediction"
                    ]
                )
            ),

        "training_seconds":
            float(
                training_seconds
            ),

        "peak_vram_mb":
            float(
                peak_vram_mb
            ),

        "embedding_dimension":
            int(
                embedding_dimension
            ),

        **params,
    }

    result_path.write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"✓ COMPLETE | "
        f"Macro-F1={best_f1:.4f} | "
        f"best_epoch={best_epoch} | "
        f"VRAM={peak_vram_mb:.0f} MB"
    )

    del model
    del optimizer
    del scaler
    del train_dataset
    del valid_dataset
    del train_loader
    del valid_loader

    gc.collect()

    torch.cuda.empty_cache()

    return result

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_n = 0

    truth = []
    predictions = []
    probabilities = []
    image_ids = []

    with torch.no_grad():

        for (
            images,
            labels,
            ids
        ) in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=AMP_DTYPE
            ):

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    labels
                )

            # IMPORTANT FIX:
            # A100 autocast uses bfloat16.
            # NumPy cannot directly convert bfloat16,
            # therefore convert probabilities to float32 first.
            probs = (
                torch.softmax(
                    logits,
                    dim=1
                )
                .float()
                .cpu()
                .numpy()
            )

            predicted = (
                logits
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

            total_loss += (
                float(
                    loss.detach()
                )
                * len(labels)
            )

            total_n += len(
                labels
            )

            truth.extend(
                labels
                .cpu()
                .tolist()
            )

            predictions.extend(
                predicted.tolist()
            )

            probabilities.extend(
                probs.tolist()
            )

            image_ids.extend(
                list(ids)
            )

    return {
        "loss":
            total_loss
            / max(
                1,
                total_n
            ),

        "truth":
            np.asarray(
                truth,
                dtype=np.int64
            ),

        "prediction":
            np.asarray(
                predictions,
                dtype=np.int64
            ),

        "probabilities":
            np.asarray(
                probabilities,
                dtype=np.float32
            ),

        "image_ids":
            np.asarray(
                image_ids
            ),
    }


# ---------------------------------------------------------------------------
# NOT in the archive: the notebook set these as Colab globals in cell 6.
# configure() sets the same names on this module so run_one() behaves exactly
# as it did there, but with the manifest, output root and seeds of a new run.
# Defaults are the archived E1-B values.
# ---------------------------------------------------------------------------
CLASSES = ["Angry", "Fear", "Happy", "Sad"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}
MODELS = ["siglip2_vit_b16", "clip_vit_b32", "efficientnet_b0"]
MODES = ["frozen", "partial", "full"]
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 16
HEAD_LR = 0.0003
ENCODER_LR = 3e-05
WEIGHT_DECAY = 0.0001
LABEL_SMOOTHING = 0.05
PARTIAL_TARGET = 0.25
NUM_WORKERS = 2
QUAL_IDS = set()


def configure(*, train_rows, valid_rows, out_root, dataset_root, manifest_sha, source_commit,
              device="cuda", **overrides):
    g = globals()
    g["train_rows"] = list(train_rows)
    g["valid_rows"] = list(valid_rows)
    g["OUT_ROOT"] = Path(out_root)
    g["DATASET_ROOT"] = Path(dataset_root)
    g["EXPECTED_T0_SHA"] = manifest_sha  # the archive stored the split manifest hash under this name
    g["SOURCE_COMMIT"] = source_commit
    g["DEVICE"] = device
    if device == "cuda" and torch.cuda.is_available():
        g["AMP_DTYPE"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        g["AMP_DTYPE"] = torch.float32
    g["CLASS_WEIGHTS"] = compute_class_weights(g["train_rows"])
    for k, v in overrides.items():
        if k not in g:
            raise KeyError(f"unknown setting {k}")
        g[k] = v
