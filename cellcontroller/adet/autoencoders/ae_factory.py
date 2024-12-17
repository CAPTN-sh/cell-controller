from dataclasses import dataclass, field
import enum
import pickle
import sklearn.exceptions
import sys
import torch
import torch.nn as nn
import warnings

from cellcontroller.adet.autoencoders.dae import LinearDAE, LstmDAE
from cellcontroller.adet.hparams import Hparams

warnings.filterwarnings("ignore", category=sklearn.exceptions.InconsistentVersionWarning)


class AEType(enum.Enum):
    LINEAR_DAE = "linear_dae"
    LSTM_DAE = "lstm_dae"


@dataclass
class DAEModsCfg:
    use_noise: bool = False
    use_sparsity: bool = False
    use_contractive: bool = False


@dataclass
class AECfg:
    ae_type: AEType = AEType.LINEAR_DAE
    hparams: Hparams = field(default_factory=lambda: Hparams())
    dae_mods: DAEModsCfg = field(default_factory=lambda: DAEModsCfg())


class AEFactory:
    @staticmethod
    def create_ae(
        cfg_or_path: AECfg | str,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> nn.Module:
        if isinstance(cfg_or_path, str):
            # HACK: temporary fix to load adet models saved with torch.save
            sys.modules["adetcsi"] = sys.modules["cellcontroller.adet"]
            m = torch.load(cfg_or_path, map_location=device, pickle_module=pickle)
            # HACK: temporary fix for the device issue with adet models saved with torch.save
            m.device = device
            return m
        else:
            ae_type = cfg_or_path.ae_type
            hparams = cfg_or_path.hparams
            dae_mods = cfg_or_path.dae_mods

            match ae_type:
                case AEType.LINEAR_DAE:
                    return LinearDAE(
                        feature_size=hparams.feature_size,
                        layer_sizes=hparams.linear_layer_sizes,
                        output_activation=hparams.output_activation,
                        use_noise=dae_mods.use_noise,
                        noise_sigma=hparams.noise_sigma,
                        noise_prob=hparams.noise_prob,
                        use_sparsity=dae_mods.use_sparsity,
                        sparsity_weight=hparams.sparsity_weight,
                        use_contractive=dae_mods.use_contractive,
                        contractive_weight=hparams.contractive_weight,
                        device=device,
                    )
                case AEType.LSTM_DAE:
                    return LstmDAE(
                        feature_size=hparams.feature_size,
                        seq_len=hparams.seq_len,
                        hidden_size=hparams.lstm_hidden_size,
                        num_layers=hparams.lstm_num_layers,
                        output_activation=hparams.output_activation,
                        use_noise=dae_mods.use_noise,
                        noise_sigma=hparams.noise_sigma,
                        noise_prob=hparams.noise_prob,
                        use_sparsity=dae_mods.use_sparsity,
                        sparsity_weight=hparams.sparsity_weight,
                        use_contractive=dae_mods.use_contractive,
                        contractive_weight=hparams.contractive_weight,
                        device=device,
                    )
