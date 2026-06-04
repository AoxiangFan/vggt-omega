import logging
import os
import random

import lightning as pl
import torch
import torch.distributed as dist
from einops import rearrange, repeat
from torch.cuda.amp import autocast
from torchvision.utils import save_image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vggt_omega.models.vggt_omega import VGGTOmega

logger = logging.getLogger(__name__)


class VGGTOmegaModel(pl.LightningModule):

    def __init__(self, opts):
        super().__init__()
        self.save_hyperparameters()
        self.run_opts = opts

        self.model = VGGTOmega(
            patch_size=opts.patch_size,
            embed_dim=opts.embed_dim,
            enable_camera=opts.enable_camera,
            enable_depth=opts.enable_depth,
            enable_alignment=opts.enable_alignment,
            use_sparse_index=opts.use_sparse_index,
            use_triton_sparse=opts.use_triton_sparse,
        )

    def step(self, phase, batch, batch_idx):
        images = batch["image_b3hw"]          # [B, V, 3, H, W], range [0, 1]
        depths = batch["depth_b1hw"][:, :, 0] # [B, V, H, W]
        intrinsics = batch["K_s0_b44"]        # [B, V, 4, 4]
        extrinsics = batch["cam_T_world_b44"] # [B, V, 4, 4]
        masks = batch["mask_b1hw"][:, :, 0]   # [B, V, H, W]

        B, V, _, H, W = images.shape

        # Normalize extrinsics relative to the first frame
        extrinsics_inv = torch.linalg.inv(extrinsics[:, 0])
        extrinsics_inv = repeat(extrinsics_inv, "B a b -> (B V) a b", V=V)
        extrinsics_flat = rearrange(extrinsics, "B V a b -> (B V) a b")
        with autocast(enabled=False):
            extrinsics_flat = extrinsics_flat @ extrinsics_inv
        extrinsics = rearrange(extrinsics_flat, "(B V) a b -> B V a b", B=B, V=V)

        masks = masks.bool() & (depths > 0)
        depths[~masks] = 0.0

        local_skip = torch.tensor([0], device=self.device)
        if (images.sum(dim=(2, 3, 4)) == 0).any():
            local_skip[0] = 1
        if dist.is_initialized():
            dist.all_reduce(local_skip, op=dist.ReduceOp.MAX)
        if bool(local_skip.item()):
            return None

        # Forward pass
        predictions = self.model(images)

        # -----------------------------------------------------------------
        # TODO: implement loss using predictions, depths, masks, intrinsics,
        #       and extrinsics.
        # predictions keys: "pose_enc", "depth", "depth_conf",
        #                   "camera_and_register_tokens"
        # -----------------------------------------------------------------
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        losses_dict = {"total_loss": loss.item()}

        if phase == "train":
            for loss_name, loss_val in losses_dict.items():
                self.log(
                    f"train/{loss_name}",
                    loss_val,
                    sync_dist=True,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )

        if phase == "val":
            self.batch_parts.append(losses_dict)

        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.batch_parts = []

    def on_validation_epoch_end(self):
        loss = [i["total_loss"] for i in self.batch_parts]
        loss = sum(loss) / len(loss)
        self.log("val/total_loss", loss, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step("val", batch, batch_idx)

    def configure_optimizers(self):
        backbone_params = []
        regular_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "patch_embed" in name:
                backbone_params.append(param)
            else:
                regular_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.run_opts.lr * self.run_opts.backbone_lr_scale},
                {"params": regular_params, "lr": self.run_opts.lr},
            ],
            weight_decay=self.run_opts.wd,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.run_opts.lr * 1e-3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
