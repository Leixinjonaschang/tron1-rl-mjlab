# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor model: 3D CNN (depth history) + MLP (proprio) → GRU → latent."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, RNN, HiddenState
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class EncoderRNNActorModel(MLPModel):
    """Actor with dual encoders feeding a GRU, plus auxiliary decoder heads.

    Forward path::

        actor obs  [B, D_actor]    → MLP proprio encoder  → [B, D_proprio]
        depth obs  [B, T, H, W]    → 3D CNN encoder       → [B, D_vision]
        cat([proprio, vision])     → GRU                  → [B, D_gru]
        cat([actor_obs, gru_out])  → actor head MLP       → actions

    Auxiliary decoder heads (used for supervised reconstruction loss)::

        gru_latent → privileged_decoder → D_privileged
        gru_latent → height_map_decoder → D_height_map

    The decoder targets are expected to live in separate obs groups (``critic``
    and ``height_map``) that are NOT in ``obs_groups["actor"]`` but ARE stored
    in rollout (because they are in ``obs_groups["critic"]``).

    Args:
        obs: Sample observation TensorDict from the environment.
        obs_groups: Mapping from obs-set names to lists of obs-group names.
        obs_set: Which obs-set this model reads (e.g. ``"actor"``).
        output_dim: Number of action dimensions.
        hidden_dims: Hidden dims of the actor head MLP.
        activation: Activation function name.
        obs_normalization: Whether to apply empirical normalisation to 1D obs.
        distribution_cfg: Config dict for the output distribution.
        actor_obs_group: Name of the 1D proprioceptive obs group.
        proprio_encoder_hidden_dims: Hidden dims of the proprioceptive MLP encoder.
        proprio_encoder_output_dim: Output dim of the proprioceptive encoder.
        depth_obs_group: Name of the depth-image obs group (shape [B, T, H, W]).
        cnn3d_output_channels: Output channel counts for each Conv3d layer.
        cnn3d_kernel_size: Kernel size shared across all Conv3d layers.
        cnn3d_strides: Stride for each Conv3d layer (scalar applied to all dims).
        cnn3d_output_dim: Output dim of the 3D CNN (after linear projection).
        rnn_hidden_dim: GRU hidden state dimension.
        rnn_num_layers: Number of GRU layers.
        privileged_decoder_obs_group: Obs group used to infer privileged target dim.
        height_map_decoder_obs_group: Obs group used to infer height-map target dim.
        decoder_hidden_dims: Hidden dims of both decoder MLPs.
    """

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        # Proprio encoder
        actor_obs_group: str = "actor",
        proprio_encoder_hidden_dims: tuple[int, ...] = (256, 128),
        proprio_encoder_output_dim: int = 64,
        # 3D CNN
        depth_obs_group: str = "depth_camera",
        cnn3d_output_channels: tuple[int, ...] = (32, 64, 64),
        cnn3d_kernel_size: int = 3,
        cnn3d_strides: tuple[int, ...] = (1, 2, 2),
        cnn3d_output_dim: int = 128,
        # GRU
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        # Decoder heads
        privileged_decoder_obs_group: str = "critic",
        height_map_decoder_obs_group: str = "height_map",
        decoder_hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        # Store before super().__init__ which calls _get_obs_dim / _get_latent_dim
        self._actor_obs_group = actor_obs_group
        self._depth_obs_group = depth_obs_group
        self._rnn_hidden_dim = rnn_hidden_dim

        super().__init__(
            obs, obs_groups, obs_set, output_dim,
            hidden_dims, activation, obs_normalization, distribution_cfg,
        )

        act_fn = resolve_nn_activation(activation)

        # --- Proprio encoder ---------------------------------------------------
        self.proprio_encoder = MLP(
            self.obs_dim, proprio_encoder_output_dim,
            proprio_encoder_hidden_dims, activation,
        )

        # --- 3D CNN ------------------------------------------------------------
        # Input shape: [B, 1, T, H, W]
        # AdaptiveAvgPool3d(1) collapses spatial + temporal dims unconditionally.
        cnn3d_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, stride in zip(cnn3d_output_channels, cnn3d_strides):
            pad = cnn3d_kernel_size // 2
            cnn3d_layers += [
                nn.Conv3d(in_ch, out_ch, kernel_size=cnn3d_kernel_size,
                          stride=stride, padding=pad),
                act_fn,
            ]
            in_ch = out_ch
        cnn3d_layers += [
            nn.AdaptiveAvgPool3d(1),     # → [B, C, 1, 1, 1]
            nn.Flatten(),                # → [B, C]
            nn.Linear(in_ch, cnn3d_output_dim),
            act_fn,
        ]
        self.cnn3d = nn.Sequential(*cnn3d_layers)

        # --- GRU ---------------------------------------------------------------
        gru_input_dim = proprio_encoder_output_dim + cnn3d_output_dim
        self.rnn = RNN(gru_input_dim, rnn_hidden_dim, rnn_num_layers, "gru")

        # --- Decoder heads -----------------------------------------------------
        privileged_dim = int(obs[privileged_decoder_obs_group].shape[-1])
        height_map_dim = int(obs[height_map_decoder_obs_group].shape[-1])
        self.privileged_decoder = MLP(rnn_hidden_dim, privileged_dim, decoder_hidden_dims, activation)
        self.height_map_decoder = MLP(rnn_hidden_dim, height_map_dim, decoder_hidden_dims, activation)

        self._last_gru_latent: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # MLPModel overrides
    # ------------------------------------------------------------------

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        """Classify each active obs group as 1D (MLP) or multi-dim (CNN), return 1D groups."""
        active_groups = obs_groups[obs_set]
        obs_groups_1d: list[str] = []
        obs_dim_1d = 0
        for group in active_groups:
            shape = obs[group].shape
            if len(shape) > 2:
                # Multi-dimensional group: only the designated depth group is expected
                if group != self._depth_obs_group:
                    raise ValueError(
                        f"Unexpected multi-dimensional obs group '{group}'. "
                        f"Only '{self._depth_obs_group}' is handled by the 3D CNN."
                    )
                continue  # handled by cnn3d
            obs_groups_1d.append(group)
            obs_dim_1d += shape[-1]
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Input to the actor head MLP = concat(actor_obs, gru_out)."""
        return self.obs_dim + self._rnn_hidden_dim

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Encode obs and return ``cat(actor_obs_normed, gru_latent)``."""
        batch_mode = masks is not None

        # Collect and normalise 1D obs (same contract as MLPModel)
        obs_1d = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
        obs_1d_normed = self.obs_normalizer(obs_1d)  # shape: [..., D_actor]

        # Depth images: [..., T_depth, H, W]
        depth = obs[self._depth_obs_group]

        if batch_mode:
            # Padded trajectories: first two dims are [T_max, B_traj]
            prefix = depth.shape[:2]  # (T_max, B_traj)

            # Proprio encoder: nn.Linear broadcasts over leading dims
            proprio_enc = self.proprio_encoder(obs_1d_normed)  # [T_max, B_traj, D_proprio]

            # 3D CNN: flatten leading dims, add channel dim, run CNN, unflatten
            depth_flat = depth.flatten(0, 1).unsqueeze(1)        # [T_max*B_traj, 1, T, H, W]
            vision_enc_flat = self.cnn3d(depth_flat)              # [T_max*B_traj, D_vision]
            vision_enc = vision_enc_flat.view(*prefix, -1)        # [T_max, B_traj, D_vision]

            # GRU: input [T_max, B_traj, D], output unpadded [B_valid, D_gru]
            gru_input = torch.cat([proprio_enc, vision_enc], dim=-1)
            gru_latent = self.rnn(gru_input, masks, hidden_state)  # [B_valid, D_gru]

            # Unpad 1D obs to match GRU output length
            actor_obs = unpad_trajectories(obs_1d_normed, masks)   # [B_valid, D_actor]
        else:
            # Rollout mode: [B, ...]
            proprio_enc = self.proprio_encoder(obs_1d_normed)       # [B, D_proprio]
            vision_enc = self.cnn3d(depth.unsqueeze(1))             # [B, D_vision]
            gru_input = torch.cat([proprio_enc, vision_enc], dim=-1)
            gru_latent = self.rnn(gru_input, None, None).squeeze(0) # [B, D_gru]
            actor_obs = obs_1d_normed

        self._last_gru_latent = gru_latent
        return torch.cat([actor_obs, gru_latent], dim=-1)

    # ------------------------------------------------------------------
    # Decoder interface (called by PPOWithDecoder)
    # ------------------------------------------------------------------

    def get_last_decoder_outputs(self) -> dict[str, torch.Tensor] | None:
        """Return decoder predictions from the most recent forward pass.

        Returns ``None`` if ``get_latent`` has not been called yet.
        """
        if self._last_gru_latent is None:
            return None
        latent = self._last_gru_latent
        return {
            "privileged": self.privileged_decoder(latent),
            "height_map": self.height_map_decoder(latent),
        }

    # ------------------------------------------------------------------
    # Recurrent interface
    # ------------------------------------------------------------------

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        return self.rnn.hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        self.rnn.detach_hidden_state(dones)
