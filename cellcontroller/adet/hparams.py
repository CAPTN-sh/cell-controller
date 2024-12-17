from dataclasses import dataclass, field
from typing import List


@dataclass
class Hparams:
    feature_size: int = 19
    seq_len: int = 8
    seq_overlap: int = 0
    output_activation: str = "sigmoid"
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    linear_layer_sizes: List[int] = field(default_factory=lambda: [128, 64, 32, 16, 8])
    noise_sigma: float = 0.5
    noise_prob: float = 0.2
    sparsity_weight: float = 1e-4
    contractive_weight: float = 1e-4
    vae_hidden_size: int = 128
    vae_latent_dim: int = 32
    rdae_l: float = 0.00065
