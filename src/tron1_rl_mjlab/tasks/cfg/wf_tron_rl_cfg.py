from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class RslRlEncoderRNNActorCfg(RslRlModelCfg):
    """Actor config for EncoderRNNActorModel.

    Architecture: 3D CNN (depth history) + MLP (proprio) → GRU → actor head.
    """

    class_name: str = "rsl_rl.models:EncoderRNNActorModel"

    # Proprioceptive MLP encoder
    actor_obs_group: str = "actor"
    proprio_encoder_hidden_dims: tuple = (256, 128)
    proprio_encoder_output_dim: int = 64

    # 3D CNN for temporal depth images
    depth_obs_group: str = "depth_camera"
    cnn3d_output_channels: tuple = (32, 64, 64)
    cnn3d_kernel_size: int = 3
    cnn3d_strides: tuple = (1, 2, 2)
    cnn3d_output_dim: int = 128

    # GRU that fuses proprio + vision encodings
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1

    # Decoder heads: reconstruct privileged obs + height map from GRU latent
    privileged_decoder_obs_group: str = "critic"
    height_map_decoder_obs_group: str = "height_map"
    decoder_hidden_dims: tuple = (256, 128)


@dataclass
class RslRlPPOWithDecoderAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with supervised decoder reconstruction loss."""

    class_name: str = "rsl_rl.algorithms:PPOWithDecoder"

    # Obs groups to use as reconstruction targets
    privileged_obs_group: str = "critic"
    height_map_obs_group: str = "height_map"

    # Weight on the total reconstruction loss (privileged + height map)
    recon_loss_coef: float = 1.0


def make_wf_tron_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for WF-TRON task."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=4000,
        save_interval=200,
        wandb_project="mjlab_wf_tron",
        experiment_name="wf_tron",
        obs_groups={
            # Actor receives current proprio + depth image history
            "actor": ("actor", "depth_camera"),
            # Critic receives privileged obs + height map (concatenated by MLPModel)
            "critic": ("critic", "height_map"),
        },
        actor=RslRlEncoderRNNActorCfg(
            hidden_dims=(256, 128),
            activation="elu",
            distribution_cfg={
                "class_name": "rsl_rl.modules:GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            class_name="rsl_rl.models:MLPModel",
            hidden_dims=(512, 256, 128),
            activation="elu",
        ),
        algorithm=RslRlPPOWithDecoderAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            recon_loss_coef=1.0,
        ),
    )
