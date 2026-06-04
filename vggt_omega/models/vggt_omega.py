# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        use_sparse_index: bool = True,
        use_triton_sparse: bool = False,
    ) -> None:
        super().__init__()

        self.aggregator = Aggregator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            use_sparse_index=use_sparse_index,
            use_triton_sparse=use_triton_sparse,
        )
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._forward(images, index=None)

    def forward_with_index(self, images: torch.Tensor, index: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass with sparse indexed attention for global inter-frame blocks.

        index: (B, V*patches, K) — for each patch token, the K positions it may
            attend to during global attention.  Values are 1-indexed; 0 is a dummy
            token used for padding.  See Aggregator.forward_with_index for details.
        """
        return self._forward(images, index=index)

    def _forward(self, images: torch.Tensor, index: torch.Tensor | None) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            if index is not None:
                aggregated_tokens_list, patch_token_start = self.aggregator.forward_with_index(images, index)
            else:
                aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
