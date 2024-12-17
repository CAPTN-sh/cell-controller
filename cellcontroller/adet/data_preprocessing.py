import numpy as np
from typing import Any, Dict, List, Literal, Tuple
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler


def extract_feature_indices(sample: Dict[str, Any], with_geo: bool = False) -> Dict[str, int]:
    feature_indices = {key: idx for idx, key in enumerate(sample.keys()) if not key.startswith("gps_")}

    # adjust for geo features if present and required
    if with_geo and "gps_lat" in sample:
        feature_indices.update(
            {
                "gps_lat": feature_indices["gps_lat"],
                "gps_lon": feature_indices["gps_lon"],
                "gps_speed": feature_indices["gps_speed"],
                "gps_heading": feature_indices["gps_heading"],
                "gps_timestamp": feature_indices["gps_timestamp"],
            }
        )

    return feature_indices


def extract_features(
    samples: List[Dict[str, Any]],
    feature_indices: Dict[str, int],
    operator: str = "vodafone",
    with_lte: bool = False,
    imputing: bool = True,
    with_geo: bool = False,
) -> np.ndarray:
    if operator not in ["vodafone", "telekom"]:
        raise ValueError("Operator must be 'vodafone' or 'telekom'")

    identity_map = (
        {"5G-D2-WAVELAB": 0, "CAU-8388": 1, "CAU-8395": 2}
        if operator == "vodafone"
        else {"5G-DTAG-WAVELAB": 0, "CAU-D4": 1, "CAU-0C": 2}
    )
    data_class_map = {"LTE": 0, "5G NSA": 1, "5G SA": 2} if with_lte else {"5G NSA": 0, "5G SA": 1}
    modulation_map = {"qpsk": 0, "16qam": 1, "64qam": 2, "256qam": 3}

    features_list = []
    for sample in samples:
        feature_vector = np.full(len(feature_indices), np.nan, dtype=np.float32)

        feature_vector[feature_indices["identity"]] = identity_map.get(sample["identity"], -1)
        feature_vector[feature_indices["phy_cell_id"]] = (
            int(sample["phy_cell_id"]) if sample["phy_cell_id"] is not None else -1
        )
        feature_vector[feature_indices["data_class"]] = data_class_map.get(sample["data_class"], -1)
        feature_vector[feature_indices["cqi"]] = (
            int(sample["cqi"]) if sample["cqi"] is not None else fill_missing_value(imputing, -1)
        )
        feature_vector[feature_indices["ri"]] = (
            int(sample["ri"]) if sample["ri"] is not None else fill_missing_value(imputing, -1)
        )
        feature_vector[feature_indices["mcs"]] = (
            int(sample["mcs"]) if sample["mcs"] is not None else fill_missing_value(imputing, -1)
        )
        feature_vector[feature_indices["modulation"]] = modulation_map.get(sample["modulation"], -1)
        feature_vector[feature_indices["rsrp"]] = (
            int(sample["rsrp"]) if sample["rsrp"] is not None else fill_missing_value(imputing, -157)
        )
        feature_vector[feature_indices["rsrq"]] = (
            int(sample["rsrq"]) if sample["rsrq"] is not None else fill_missing_value(imputing, -100)
        )
        feature_vector[feature_indices["sinr"]] = (
            int(sample["sinr"]) if sample["sinr"] is not None else fill_missing_value(imputing, -24)
        )
        feature_vector[feature_indices["rx_packets"]] = int(sample["rx_packets"])
        feature_vector[feature_indices["rx_bits"]] = int(sample["rx_bits"])
        feature_vector[feature_indices["rx_drops"]] = int(sample["rx_drops"])
        feature_vector[feature_indices["rx_errors"]] = int(sample["rx_errors"])
        feature_vector[feature_indices["tx_packets"]] = int(sample["tx_packets"])
        feature_vector[feature_indices["tx_bits"]] = int(sample["tx_bits"])
        feature_vector[feature_indices["tx_drops"]] = int(sample["tx_drops"])
        feature_vector[feature_indices["tx_queue_drops"]] = int(sample["tx_queue_drops"])
        feature_vector[feature_indices["tx_errors"]] = int(sample["tx_errors"])

        if with_geo:
            feature_vector[feature_indices["gps_lat"]] = float(sample["gps_lat"])
            feature_vector[feature_indices["gps_lon"]] = float(sample["gps_lon"])
            feature_vector[feature_indices["gps_speed"]] = float(str(sample["gps_speed"]).split()[0])
            feature_vector[feature_indices["gps_heading"]] = float(str(sample["gps_heading"]).split()[0])

        features_list.append(feature_vector)

    features_np = np.array(features_list, dtype=np.float32)
    return features_np


def preprocess_features(
    samples: List[Dict[str, Any]],
    feature_indices: Dict[str, int],
    imputer: IterativeImputer,
    encoder: OrdinalEncoder,
    operator: str = "vodafone",
    with_lte: bool = True,
) -> np.ndarray:
    features = extract_features(samples, feature_indices, operator, with_lte, imputer is not None, False)

    if imputer is not None and np.isnan(features).any():
        features = imputer.transform(features)

    phy_cell_id_idx = feature_indices["phy_cell_id"]
    features[:, phy_cell_id_idx] = encoder.transform(features[:, phy_cell_id_idx].reshape(-1, 1)).flatten()

    return features


def filter_technical_anomalies(
    features: np.ndarray,
    feature_indices: Dict[str, int],
    decrement_lower: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    sinr_idx = feature_indices["sinr"]
    rsrp_idx = feature_indices["rsrp"]
    cqi_idx = feature_indices["cqi"]
    ri_idx = feature_indices["ri"]
    mcs_idx = feature_indices["mcs"]

    decrement = 1 if decrement_lower else 0

    # NOTE: the lower bound for SINR, RSRP, CQI, RI, and MCS are lowered by 1 from the 3GPP standard
    #       to enable possible filling of missing values with -1 of the lower bound
    valid_mask = (
        (features[:, sinr_idx] >= -23 - decrement)
        & (features[:, sinr_idx] <= 40)
        & (features[:, rsrp_idx] >= -156 - decrement)
        & (features[:, rsrp_idx] <= -31)
        & (features[:, cqi_idx] >= 0 - decrement)
        & (features[:, cqi_idx] <= 15)
        & (features[:, ri_idx] >= 0 - decrement)
        & (features[:, ri_idx] <= 4)
        & (features[:, mcs_idx] >= 0 - decrement)
        & (features[:, mcs_idx] <= 31)
    )

    valid_features = features[valid_mask]
    invalid_features = features[~valid_mask]

    return valid_features, invalid_features


def apply_preprocessing_for_inference(
    samples: List[Dict[str, Any]],
    feature_indices: Dict[str, int],
    imputer: IterativeImputer | None,
    encoder: OrdinalEncoder,
    operator: str = "vodafone",
    with_lte: bool = True,
) -> Tuple[np.ndarray, int]:
    features = preprocess_features(samples, feature_indices, imputer, encoder, operator, with_lte)
    features, tas = filter_technical_anomalies(features, feature_indices, imputer is None)
    return features, tas.size


def scale_test_data(test_data: np.ndarray, scaler: StandardScaler | MinMaxScaler) -> np.ndarray:
    return scaler.transform(test_data)


def unscale_test_data(data: np.ndarray, scaler: StandardScaler | MinMaxScaler) -> np.ndarray:
    return scaler.inverse_transform(data)


def pack_into_sequences(
    data: np.ndarray,
    seq_len: int,
    overlap: int = 0,
    padding_type: Literal['pre', 'post'] = 'post',  # def as in pytorch
) -> np.ndarray:
    overlap = max(0, min(overlap, seq_len))
    padded_data = []
    num_samples, num_features = data.shape
    idx = 0
    while idx < num_samples:
        end_idx = min(idx + seq_len, num_samples)
        sequence = data[idx:end_idx]
        if len(sequence) < seq_len:
            pad_size = seq_len - len(sequence)
            pad_width = ((pad_size, 0), (0, 0)) if padding_type == 'pre' else ((0, pad_size), (0, 0))
            sequence = np.pad(sequence, pad_width, mode="constant", constant_values=0)
        padded_data.append(sequence)
        idx += seq_len - overlap
    padded_data = np.array(padded_data)
    return padded_data.reshape(-1, seq_len, num_features)


def round_with_tolerance(arr: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    rounded_arr = np.round(arr)
    return np.where(np.isclose(arr, rounded_arr, atol=tolerance), rounded_arr, arr)


def fill_missing_value(imputing: bool = True, replacement_value: Any | None = None) -> Any:
    if imputing:
        return np.nan
    else:
        if replacement_value is None:
            return np.nan
        else:
            return replacement_value
