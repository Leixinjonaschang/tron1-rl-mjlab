# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor model: 2D CNN (current depth frame) + MLP (proprio) → GRU → latent.

The CNN runs with no_grad for the GRU forward pass so the PPO and GRU-decoder
gradients do NOT flow through the CNN.  A second CNN forward pass (with grad)
is triggered inside get_last_decoder_outputs() to build the CNN-decoder loss
computation graph separately.

This avoids storing the full Conv2d activation volume for all T_rollout × B_mini
samples simultaneously — the no_grad pass discards activations immediately, and
the with-grad pass only materialises activations for the decoder loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, RNN, HiddenState
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class EncoderRNNActorModel(MLPModel):
    """Actor with dual encoders feeding a GRU, plus four auxiliary decoder heads.

    Forward path::

        actor obs  [B, D_actor]  → MLP proprio encoder  → [B, D_proprio]
        depth obs  [B, H, W]     → 2D CNN (no_grad)      → [B, D_vision]
        cat([proprio, vision])   → GRU                  → [B, D_gru]
        cat([actor_obs, gru_out]) → actor head MLP       → actions

    Two independent sets of decoder heads (Option B)::

        gru_latent → privileged_decoder     → D_privileged  (gradient: GRU + decoder)
        gru_latent → height_map_decoder     → D_height_map  (gradient: GRU + decoder)
        cnn_feat   → cnn_privileged_decoder → D_privileged  (gradient: CNN + decoder)
        cnn_feat   → cnn_height_map_decoder → D_height_map  (gradient: CNN + decoder)

    The CNN decoder path is built by re-running the CNN with grad inside
    ``get_last_decoder_outputs()``.  The depth input from the most recent
    ``get_latent()`` call is cached for this purpose.

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
        depth_obs_group: Name of the depth-image obs group (shape [B, H, W]).
        cnn_output_channels: Output channel counts for each Conv2d layer.
        cnn_kernel_size: Kernel size shared across all Conv2d layers.
        cnn_strides: Stride for each Conv2d layer.
        cnn_output_dim: Output dim of the 2D CNN (after linear projection).
        rnn_hidden_dim: GRU hidden state dimension.
        rnn_num_layers: Number of GRU layers.
        privileged_decoder_obs_group: Obs group used to infer privileged target dim.
        height_map_decoder_obs_group: Obs group used to infer height-map target dim.
        decoder_hidden_dims: Hidden dims shared by all four decoder MLPs.
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
        # 2D CNN
        depth_obs_group: str = "depth_camera",
        cnn_output_channels: tuple[int, ...] = (32, 64, 64),
        cnn_kernel_size: int = 3,
        cnn_strides: tuple[int, ...] = (2, 2, 2),
        cnn_output_dim: int = 128,
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

        # --- 2D CNN ------------------------------------------------------------
        # Input: [B, 1, H, W]  (channel dim added by unsqueeze before passing)
        # All strides default to 2 so spatial dims halve at each layer.
        cnn_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, stride in zip(cnn_output_channels, cnn_strides):
            pad = cnn_kernel_size // 2
            cnn_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=cnn_kernel_size,
                          stride=stride, padding=pad),
                act_fn,
            ]
            in_ch = out_ch
        cnn_layers += [
            nn.AdaptiveAvgPool2d(1),  # → [B, C, 1, 1]
            nn.Flatten(),             # → [B, C]
            nn.Linear(in_ch, cnn_output_dim),
            act_fn,
        ]
        self.cnn2d = nn.Sequential(*cnn_layers)

        # --- GRU ---------------------------------------------------------------
        gru_input_dim = proprio_encoder_output_dim + cnn_output_dim
        self.rnn = RNN(gru_input_dim, rnn_hidden_dim, rnn_num_layers, "gru")

        # --- GRU decoder heads (gradient flows to GRU + these decoders) --------
        privileged_dim = int(obs[privileged_decoder_obs_group].shape[-1])
        height_map_dim = int(obs[height_map_decoder_obs_group].shape[-1])
        self.privileged_decoder = MLP(rnn_hidden_dim, privileged_dim, decoder_hidden_dims, activation)
        self.height_map_decoder = MLP(rnn_hidden_dim, height_map_dim, decoder_hidden_dims, activation)

        # --- CNN decoder heads (gradient flows to CNN + these decoders) ---------
        self.cnn_privileged_decoder = MLP(cnn_output_dim, privileged_dim, decoder_hidden_dims, activation)
        self.cnn_height_map_decoder = MLP(cnn_output_dim, height_map_dim, decoder_hidden_dims, activation)

        self._last_gru_latent: torch.Tensor | None = None
        self._last_depth_input: torch.Tensor | None = None
        self._last_batch_mode: bool = False

    # ------------------------------------------------------------------
    # MLPModel overrides
    # ------------------------------------------------------------------

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        """Classify each active obs group as 1D (MLP) or image (CNN); return 1D groups."""
        active_groups = obs_groups[obs_set]
        obs_groups_1d: list[str] = []
        obs_dim_1d = 0
        for group in active_groups:
            shape = obs[group].shape
            if len(shape) > 2:
                if group != self._depth_obs_group:
                    raise ValueError(
                        f"Unexpected multi-dimensional obs group '{group}'. "
                        f"Only '{self._depth_obs_group}' is handled by the 2D CNN."
                    )
                continue  # handled by cnn2d
            obs_groups_1d.append(group)
            obs_dim_1d += shape[-1]
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self._rnn_hidden_dim

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Encode obs and return ``cat(actor_obs_normed, gru_latent)``."""
        batch_mode = masks is not None
        self._last_batch_mode = batch_mode

        # 1D obs
        obs_1d = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
        obs_1d_normed = self.obs_normalizer(obs_1d)

        # Depth: [B, H, W] in rollout mode; [T_max, B_mini, H, W] in batch mode
        depth = obs[self._depth_obs_group]
        self._last_depth_input = depth  # cache for CNN decoder re-run (with grad)

        if batch_mode:
            prefix = depth.shape[:2]  # (T_max, B_mini)

            # Proprio: nn.Linear broadcasts over leading dims
            proprio_enc = self.proprio_encoder(obs_1d_normed)  # [T, B, D_proprio]

            # CNN — no_grad so PPO loss does not flow through CNN
            with torch.no_grad():
                depth_flat = depth.flatten(0, 1).unsqueeze(1)       # [T*B, 1, H, W]
                vision_enc_flat = self.cnn2d(depth_flat)             # [T*B, D_vision]
            vision_enc = vision_enc_flat.detach().view(*prefix, -1)  # [T, B, D_vision]

            # GRU: outputs unpadded [B_valid, D_gru]
            gru_input = torch.cat([proprio_enc, vision_enc], dim=-1)
            gru_latent = self.rnn(gru_input, masks, hidden_state)

            # Unpad 1D obs to match GRU output length
            actor_obs = unpad_trajectories(obs_1d_normed, masks)
        else:
            # Rollout mode: [B, ...]
            proprio_enc = self.proprio_encoder(obs_1d_normed)        # [B, D_proprio]
            with torch.no_grad():
                vision_enc = self.cnn2d(depth.unsqueeze(1))          # [B, D_vision]
            vision_enc = vision_enc.detach()
            gru_input = torch.cat([proprio_enc, vision_enc], dim=-1)
            gru_latent = self.rnn(gru_input, None, None).squeeze(0)  # [B, D_gru]
            actor_obs = obs_1d_normed

        self._last_gru_latent = gru_latent
        return torch.cat([actor_obs, gru_latent], dim=-1)

    # ------------------------------------------------------------------
    # Decoder interface (called by PPOWithDecoder)
    # ------------------------------------------------------------------

    def get_last_decoder_outputs(self, masks: torch.Tensor | None = None) -> dict[str, torch.Tensor] | None:
        """Return all four decoder predictions.

        GRU decoder outputs use the cached ``_last_gru_latent`` (already
        unpadded by the GRU module).

        CNN decoder outputs re-run the CNN **with grad** on the cached depth
        input so the CNN parameters receive gradients from the CNN decoder loss.
        The CNN features are unpadded using ``masks`` when in batch mode.

        Returns ``None`` if ``get_latent`` has not been called yet.
        """
        if self._last_gru_latent is None:
            return None

        outputs: dict[str, torch.Tensor] = {}

        # GRU decoder heads
        outputs["privileged"] = self.privileged_decoder(self._last_gru_latent)
        outputs["height_map"] = self.height_map_decoder(self._last_gru_latent)

        # CNN decoder heads — re-run CNN with gradient
        if self._last_depth_input is not None:
            depth = self._last_depth_input
            if self._last_batch_mode:
                prefix = depth.shape[:2]  # (T_max, B_mini)
                depth_flat = depth.flatten(0, 1).unsqueeze(1)       # [T*B, 1, H, W]
                cnn_feat = self.cnn2d(depth_flat)                   # [T*B, D_vision]
                cnn_feat = cnn_feat.view(*prefix, -1)               # [T, B, D_vision]
                if masks is not None:
                    cnn_feat = unpad_trajectories(cnn_feat, masks)  # [B_valid, D_vision]
                else:
                    cnn_feat = cnn_feat.flatten(0, 1)
            else:
                cnn_feat = self.cnn2d(depth.unsqueeze(1))           # [B, D_vision]

            outputs["cnn_privileged"] = self.cnn_privileged_decoder(cnn_feat)
            outputs["cnn_height_map"] = self.cnn_height_map_decoder(cnn_feat)

        return outputs

    # ------------------------------------------------------------------
    # Recurrent interface
    # ------------------------------------------------------------------

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        return self.rnn.hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        self.rnn.detach_hidden_state(dones)
