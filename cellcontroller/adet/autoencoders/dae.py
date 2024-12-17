from datetime import datetime
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple
from tqdm import tqdm

from cellcontroller.adet.autoencoders.utils import custom_decay_lr
from cellcontroller.adet.data_preprocessing import round_with_tolerance, unscale_test_data


class DAE(nn.Module):
    def __init__(
        self,
        feature_size: int,
        output_activation: str = "sigmoid",
        use_noise: bool = False,
        noise_sigma: float = 0.5,
        noise_prob: float = 0.1,
        use_sparsity: bool = False,
        sparsity_weight: float = 1e-5,
        use_contractive: bool = False,
        contractive_weight: float = 1e-5,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.output_activation = output_activation if output_activation in ["sigmoid", "relu"] else "sigmoid"
        self.use_noise = use_noise
        self.noise_sigma = noise_sigma
        self.noise_prob = noise_prob
        self.use_sparsity = use_sparsity
        self.sparsity_weight = sparsity_weight
        self.use_contractive = use_contractive
        self.contractive_weight = contractive_weight

        self.device = device

        self.encoder = None
        self.decoder = None

        self.threshold = None
        self.epochs = 0

    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method")

    def apply_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.noise_sigma * torch.randn_like(x)
        mask = np.random.uniform(size=(x.shape))
        mask = (mask < self.noise_prob).astype(int)
        noise *= torch.tensor(mask).to(self.device)
        return x + noise

    def sparsity_reg(self) -> torch.Tensor:
        sparsity_loss = 0
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                sparsity_loss += torch.mean(torch.abs(layer.weight)).to(self.device)
        term = self.sparsity_weight * sparsity_loss
        return term

    def contractive_reg(self) -> torch.Tensor:
        enc_weights = [layer.weight for layer in self.encoder if isinstance(layer, nn.Linear)]
        jacobian = torch.cat([weight.view(-1) for weight in enc_weights])
        term = self.contractive_weight * torch.norm(jacobian, p="fro").to(self.device)
        return term


class LstmDAE(DAE):
    def __init__(
        self,
        feature_size: int,
        seq_len: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        is_xavier_init: bool = True,
        output_activation: str = "sigmoid",
        use_noise: bool = False,
        noise_sigma: float = 0.5,
        noise_prob: float = 0.1,
        use_sparsity: bool = False,
        sparsity_weight: float = 1e-5,
        use_contractive: bool = False,
        contractive_weight: float = 1e-5,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__(
            feature_size,
            output_activation,
            use_noise,
            noise_sigma,
            noise_prob,
            use_sparsity,
            sparsity_weight,
            use_contractive,
            contractive_weight,
            device,
        )

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_xavier_init = is_xavier_init

        self.encoder = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, feature_size)
        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "relu":
            self.output_activation = nn.ReLU()
        else:
            self.output_activation = nn.Tanh()

    def forward(self, x) -> torch.Tensor:
        if self.use_noise:
            x = self.apply_noise(x)

        batch_size = x.shape[0]

        h_enc, c_enc = self._init_hidden(batch_size, self.hidden_size, self.is_xavier_init)
        _, (h_enc, c_enc) = self.encoder(x, (h_enc, c_enc))

        h_dec, c_dec = self._init_hidden(batch_size, self.hidden_size, self.is_xavier_init)
        decoder_input = h_enc[-1].unsqueeze(1).repeat(1, self.seq_len, 1)

        output, _ = self.decoder(decoder_input, (h_dec, c_dec))
        output = self.linear(output)
        output = self.output_activation(output)

        return output

    def _init_hidden(self, batch_size: int, hidden_size: int, is_xavier: bool = True) -> torch.Tensor:
        if is_xavier:
            h = nn.init.xavier_normal_(torch.empty(self.num_layers, batch_size, hidden_size, device=self.device))
            c = nn.init.xavier_normal_(torch.empty(self.num_layers, batch_size, hidden_size, device=self.device))
        h = torch.zeros(self.num_layers, batch_size, hidden_size).to(self.device)
        c = torch.zeros(self.num_layers, batch_size, hidden_size).to(self.device)
        return h, c

    def sparsity_reg(self) -> torch.Tensor:
        sparsity_loss = 0

        for layer_idx in range(self.encoder.num_layers):
            weight_ih = getattr(self.encoder, f'weight_ih_l{layer_idx}')
            weight_hh = getattr(self.encoder, f'weight_hh_l{layer_idx}')
            bias_ih = getattr(self.encoder, f'bias_ih_l{layer_idx}', None)
            bias_hh = getattr(self.encoder, f'bias_hh_l{layer_idx}', None)
            sparsity_loss += torch.mean(torch.abs(weight_ih)).to(self.device)
            sparsity_loss += torch.mean(torch.abs(weight_hh)).to(self.device)
            if bias_ih is not None:
                sparsity_loss += torch.mean(torch.abs(bias_ih)).to(self.device)
            if bias_hh is not None:
                sparsity_loss += torch.mean(torch.abs(bias_hh)).to(self.device)

        term = self.sparsity_weight * sparsity_loss
        return term

    def contractive_reg(self) -> torch.Tensor:
        weights = []
        biases = []

        for layer_idx in range(self.encoder.num_layers):
            weight_ih = getattr(self.encoder, f'weight_ih_l{layer_idx}')
            weight_hh = getattr(self.encoder, f'weight_hh_l{layer_idx}')
            bias_ih = getattr(self.encoder, f'bias_ih_l{layer_idx}', None)
            bias_hh = getattr(self.encoder, f'bias_hh_l{layer_idx}', None)
            weights.append(weight_ih.view(-1))
            weights.append(weight_hh.view(-1))
            if bias_ih is not None:
                biases.append(bias_ih.view(-1))
            if bias_hh is not None:
                biases.append(bias_hh.view(-1))

        jacobian = torch.cat(weights + biases)
        term = self.contractive_weight * torch.norm(jacobian, p="fro").to(self.device)
        return term


class LinearDAE(DAE):
    def __init__(
        self,
        feature_size: int,
        layer_sizes: List[int],
        output_activation: str = "sigmoid",
        use_noise: bool = False,
        noise_sigma: float = 0.5,
        noise_prob: float = 0.1,
        use_sparsity: bool = False,
        sparsity_weight: float = 1e-5,
        use_contractive: bool = False,
        contractive_weight: float = 1e-5,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__(
            feature_size,
            output_activation,
            use_noise,
            noise_sigma,
            noise_prob,
            use_sparsity,
            sparsity_weight,
            use_contractive,
            contractive_weight,
            device,
        )

        layers = []
        in_size = feature_size
        for out_size in layer_sizes:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            in_size = out_size
        self.encoder = nn.Sequential(*layers)

        layers = []
        for out_size in reversed(layer_sizes[:-1]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            in_size = out_size
        layers.append(nn.Linear(in_size, feature_size))
        layers.append(nn.Sigmoid() if self.output_activation == "sigmoid" else nn.ReLU())
        self.decoder = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        if self.use_noise:
            x = self.apply_noise(x)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def make_dae_model_name(model: DAE) -> str:
    model_name = model.__class__.__name__.lower() + "_"
    if model.use_noise:
        model_name += "n_"
    if model.use_sparsity:
        model_name += "s_"
    if model.use_contractive:
        model_name += "c_"
    model_name += datetime.now().strftime("%Y-%m-%d-%H_%M_%S_%f")[:-3]
    return model_name


def train_dae(
    X_train: np.ndarray,
    X_vali: np.ndarray | None,
    model: DAE,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    decay_learning_rate_type: str | None = None,
    threshold_percentile: float = 95,
    patience: int = 5,
    shuffle: bool = False,
    save_path: str | None = None,
    device: str = "cpu",
) -> Tuple[nn.Module, List[float], float]:
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    dataset = TensorDataset(X_train_tensor, X_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    if X_vali is not None:
        X_vali_tensor = torch.FloatTensor(X_vali).to(device)

    model = model.to(device)
    init_lr = learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss(reduction="mean")

    loss_values = []
    vali_loss = float('inf')
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
        for inputs, _ in progress_bar:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            train_loss = criterion(outputs, inputs)
            if model.use_sparsity:
                train_loss += model.sparsity_reg()
            if model.use_contractive:
                train_loss += model.contractive_reg()
            train_loss.backward()

            optimizer.step()

            epoch_losses.append(train_loss.item())
            progress_bar.set_postfix({"loss": np.mean(epoch_losses), "vali_loss": vali_loss})

        loss_values.extend(epoch_losses)

        if X_vali is not None:
            with torch.no_grad():
                model.eval()
                output = model(X_vali_tensor)
                vali_loss = criterion(output, X_vali_tensor).mean().item()
            if vali_loss - best_loss < -0.01:
                best_loss = vali_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}..")
                    break

        if decay_learning_rate_type:
            custom_decay_lr(decay_learning_rate_type, optimizer, init_lr, epoch + 1, epochs)

    model.threshold = np.percentile(np.array(loss_values), threshold_percentile)
    model.epochs = epoch + 1

    if save_path:
        torch.save(model, os.path.join(save_path, make_dae_model_name(model) + ".pt"))

    return model, loss_values, best_loss


def evaluate_dae(
    X_test: np.ndarray,
    model: DAE,
    threshold: float,
    scaler: StandardScaler | MinMaxScaler | None = None,
    device: str = "cpu",
    save_path: str | None = None,
    columns: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model = model.to(device)
    model.eval()

    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        reconstructed_outputs = model(X_test_tensor)

    criterion = nn.MSELoss(reduction="none")
    if len(X_test_tensor.shape) == 3:
        loss_per_sample = criterion(reconstructed_outputs, X_test_tensor).mean(dim=(1, 2)).cpu().numpy()
    else:
        loss_per_sample = criterion(reconstructed_outputs, X_test_tensor).mean(dim=1).cpu().numpy()

    anomalies_mask = loss_per_sample > threshold
    anomalies_indices = np.where(anomalies_mask)[0].tolist()

    anomalies_arr = X_test[anomalies_indices]
    if len(X_test_tensor.shape) == 3:
        anomalies_arr = anomalies_arr.reshape(-1, anomalies_arr.shape[2])

    score = (len(anomalies_indices) / len(X_test)) * 100

    if anomalies_arr.size > 0:
        if scaler:
            anomalies_arr = unscale_test_data(anomalies_arr, scaler)

        anomalies_arr = round_with_tolerance(anomalies_arr, 1e-5)

        if save_path:
            np.savetxt(
                os.path.join(save_path, f"a_{make_dae_model_name(model)}.csv"),
                anomalies_arr,
                fmt="%d",
                delimiter=",",
                header=",".join(columns) if columns else "",
                comments="",
            )

    return anomalies_arr, loss_per_sample, score
