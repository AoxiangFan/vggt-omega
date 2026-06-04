import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy, Strategy
from torch.utils.data import DataLoader, ConcatDataset

_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_VGGT_OMEGA_ROOT = os.path.join(_TRAIN_DIR, "..")

sys.path.insert(0, _VGGT_OMEGA_ROOT)
sys.path.insert(0, _TRAIN_DIR)

import options as options
torch.serialization.add_safe_globals([options.Options, options.DataOptions])

from vggt_omega_model import VGGTOmegaModel
from utils.dataset_utils import get_dataset


def prepare_dataloaders(opts: options.Options) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders from opts."""
    train_datasets, val_datasets = [], []

    for dataset in opts.datasets:
        dataset_class, scans = get_dataset(
            dataset.dataset, dataset.dataset_scan_split_file, opts.single_debug_scan_id
        )
        train_datasets.append(
            dataset_class(
                dataset.dataset_path,
                split="train",
                mv_tuple_file_suffix=dataset.mv_tuple_file_suffix,
                num_images_in_tuple=opts.num_images_in_tuple,
                tuple_info_file_location=dataset.tuple_info_file_location,
                image_width=opts.image_width,
                image_height=opts.image_height,
                shuffle_tuple=opts.shuffle_tuple,
                scans=scans,
                single_output_format=opts.single_output_format,
            )
        )

    for dataset in opts.val_datasets:
        dataset_class, scans = get_dataset(
            dataset.dataset, dataset.dataset_scan_split_file, opts.single_debug_scan_id
        )

        val_datasets.append(
            dataset_class(
                dataset.dataset_path,
                split="val",
                mv_tuple_file_suffix=dataset.mv_tuple_file_suffix,
                num_images_in_tuple=opts.num_images_in_tuple,
                tuple_info_file_location=dataset.tuple_info_file_location,
                image_width=opts.val_image_width,
                image_height=opts.val_image_height,
                include_full_res_depth=opts.high_res_validation,
                scans=scans,
                single_output_format=opts.single_output_format,
            )
        )

    train_dataloader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=opts.num_workers > 0,
    )

    val_dataloader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=opts.val_batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=opts.num_workers > 0,
    )

    return train_dataloader, val_dataloader


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def prepare_model(opts: options.Options) -> VGGTOmegaModel:
    """Instantiate the model and optionally load weights."""

    if opts.load_weights_from_checkpoint is not None:
        model = VGGTOmegaModel.load_from_checkpoint(opts.load_weights_from_checkpoint)

    elif opts.lazy_load_weights_from_checkpoint is not None:
        model = VGGTOmegaModel(opts)
        ckpt = torch.load(opts.lazy_load_weights_from_checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        # Use strict=False because we may have added new parameters (dummy_k/dummy_v)
        # that are absent from the pre-trained checkpoint.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    else:
        model = VGGTOmegaModel(opts)

    return model


# ---------------------------------------------------------------------------
# Trainer helpers
# ---------------------------------------------------------------------------

def prepare_callbacks(opts: options.Options) -> List[pl.pytorch.callbacks.Callback]:
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor="val/total_loss",
        mode="min",
        dirpath=str((Path(opts.log_dir) / opts.name).resolve()),
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    return [checkpoint_callback, lr_monitor]


def prepare_ddp_strategy(opts: options.Options) -> Strategy:
    return DDPStrategy(find_unused_parameters=True)


def prepare_trainer(
    opts: options.Options,
    logger,
    callbacks: List[pl.pytorch.callbacks.Callback],
    ddp_strategy: Strategy,
) -> pl.Trainer:
    return pl.Trainer(
        devices=opts.gpus,
        log_every_n_steps=opts.log_interval,
        val_check_interval=opts.val_interval,
        limit_val_batches=opts.val_batches,
        max_epochs=opts.max_epochs,
        precision=opts.precision,
        benchmark=True,
        logger=logger,
        sync_batchnorm=False,
        callbacks=callbacks,
        num_sanity_val_steps=opts.num_sanity_val_steps,
        gradient_clip_val=opts.grad_clip_norm,
        strategy=ddp_strategy,
        limit_train_batches=10000,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(opts):
    pl.seed_everything(opts.random_seed)

    folder = os.path.join(opts.log_dir, opts.name)
    os.makedirs(folder, exist_ok=True)

    model = prepare_model(opts)
    train_dataloader, val_dataloader = prepare_dataloaders(opts)

    logger = WandbLogger(
        project="vggt_omega",
        name=opts.name,
        save_dir=opts.log_dir,
        offline=opts.debug,
    )

    ddp_strategy = prepare_ddp_strategy(opts)
    callbacks = prepare_callbacks(opts)

    trainer = prepare_trainer(
        opts=opts,
        logger=logger,
        callbacks=callbacks,
        ddp_strategy=ddp_strategy,
    )

    resume_ckpt = opts.resume if opts.resume is not None else "last"
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print()
    opts = option_handler.options

    if opts.gpus == 0:
        print("Setting precision to 32 since --gpus is 0.")
        opts.precision = 32

    main(opts)
