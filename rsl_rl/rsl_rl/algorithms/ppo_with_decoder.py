# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO extended with supervised reconstruction losses from GRU latent."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories


class PPOWithDecoder(PPO):
    """PPO + MSE reconstruction losses for privileged obs and height map.

    After each actor forward pass the GRU latent is decoded into two
    predictions.  The mean-squared error between each prediction and the
    corresponding stored observation target is added to the PPO loss::

        L_total = L_ppo
                + recon_loss_coef * MSE(decode_privileged(h), obs["critic"])
                + recon_loss_coef * MSE(decode_height_map(h),  obs["height_map"])

    The targets are read directly from ``batch.observations`` (all obs groups
    are stored in rollout regardless of which groups the models actively use).
    In the recurrent mini-batch case the targets are unpadded to match the
    unpadded GRU output.

    Args:
        privileged_obs_group: Key in ``batch.observations`` used as the
            reconstruction target for the privileged decoder.
        height_map_obs_group: Key in ``batch.observations`` used as the
            reconstruction target for the height-map decoder.
        recon_loss_coef: Weight applied to the total reconstruction loss.
    """

    def __init__(
        self,
        actor,
        critic,
        storage: RolloutStorage,
        privileged_obs_group: str = "critic",
        height_map_obs_group: str = "height_map",
        recon_loss_coef: float = 1.0,
        **ppo_kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **ppo_kwargs)
        self.privileged_obs_group = privileged_obs_group
        self.height_map_obs_group = height_map_obs_group
        self.recon_loss_coef = recon_loss_coef

    def _compute_auxiliary_loss(
        self, batch: RolloutStorage.Batch, original_batch_size: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute MSE reconstruction loss from the actor's decoder heads."""
        decoder_outputs = self.actor.get_last_decoder_outputs()  # type: ignore[attr-defined]
        if decoder_outputs is None:
            return torch.zeros(1, device=self.device).squeeze(), {}

        priv_pred = decoder_outputs["privileged"]
        hmap_pred = decoder_outputs["height_map"]

        # Unpad observation targets if we're in recurrent (padded trajectory) mode
        if batch.masks is not None:
            priv_target = unpad_trajectories(
                batch.observations[self.privileged_obs_group], batch.masks  # type: ignore[index]
            ).detach()
            hmap_target = unpad_trajectories(
                batch.observations[self.height_map_obs_group], batch.masks  # type: ignore[index]
            ).detach()
        else:
            priv_target = batch.observations[self.privileged_obs_group].detach()  # type: ignore[index]
            hmap_target = batch.observations[self.height_map_obs_group].detach()  # type: ignore[index]

        priv_loss = F.mse_loss(priv_pred, priv_target)
        hmap_loss = F.mse_loss(hmap_pred, hmap_target)
        aux_loss = self.recon_loss_coef * (priv_loss + hmap_loss)

        return aux_loss, {
            "recon_privileged": priv_loss.item(),
            "recon_height_map": hmap_loss.item(),
        }
