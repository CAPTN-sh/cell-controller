from datetime import datetime
from enum import Enum
import numpy as np
import os
import pickle
from sklearn.neighbors import BallTree
from typing import Any, Dict, List, Self

from cellcontroller.structs import CellReport

NEIGHBOUR_RADIUS_KM = 0.5
APPROX_EARTH_RADIUS_KM = 6371.0


class CellEvent(Enum):
    CELL_SWITCH = "cell_switch"
    NET_SWITCH = "network_switch"
    SIG_FLUCT = "signal_fluctuation"
    RSRP_SINR_ASYMM = "rsrp_sinr_asymmetry"
    DATA_ERR = "data_error"
    ANOM_HOUR = "hourly_anomaly"
    ANOM_POS = "position_anomaly"
    ANOM_POS_INTENSITY = "position_anomaly_intensity"
    ANOM_NAV = "navigation_anomaly"
    DATA_BURST = "data_burst"
    DATA_THROT = "data_throttle"


class CellEventClassifier:
    def __init__(self, baselines_file_path: str | None):
        self.baselines = {}
        self.prev_sample = None

        if baselines_file_path:
            self._load_baselines(baselines_file_path)
        # FIXME: handle the case when baselines_file_path is None

    def identify_events(self, data: Dict[str, Any] | List[Dict[str, Any]]) -> List[str]:
        if isinstance(data, dict):
            return self._process_point(data)
        elif isinstance(data, list):
            return self._process_sequence(data)
        else:
            raise TypeError(f'Invalid data type: {type(data)}')

    def _process_point(self, sample: Dict[str, Any]) -> List[str]:
        events = []

        if self.prev_sample:
            # CELL_SWITCH
            if sample['phy_cell_id'] != self.prev_sample['phy_cell_id']:
                events.append(
                    CellEvent.CELL_SWITCH.name + f"_{self.prev_sample['phy_cell_id']}_to_{sample['phy_cell_id']}"
                )

            # NET_SWITCH
            if sample['data_class'] != self.prev_sample['data_class']:
                events.append(
                    CellEvent.NET_SWITCH.name + f"_from_{self.prev_sample['data_class']}_to_{sample['data_class']}"
                )

            # SIG_FLUCT
            for feature in ['rsrp', 'sinr', 'cqi']:
                std_dev = self.baselines.get(f'{feature}_std', 0)
                if std_dev > 0:
                    diff = abs(sample[feature] - self.prev_sample[feature])
                    if diff > 2 * std_dev:
                        events.append(CellEvent.SIG_FLUCT.name + f"_{feature}")

            # RSRP_SINR_ASYMM
            sinr_std = self.baselines.get('sinr_std', 0)
            rsrp_std = self.baselines.get('rsrp_std', 0)
            if sinr_std > 0 and rsrp_std > 0:
                sinr_diff = sample['sinr'] - self.prev_sample['sinr']
                rsrp_diff = sample['rsrp'] - self.prev_sample['rsrp']
                sinr_sig_change = abs(sinr_diff) > 2 * sinr_std
                rsrp_sig_change = abs(rsrp_diff) > 2 * rsrp_std

                if (sinr_sig_change and not rsrp_sig_change) or (rsrp_sig_change and not sinr_sig_change):
                    events.append(CellEvent.RSRP_SINR_ASYMM.name)

            # DATA_ERR
            tx_errors_diff = sample['tx_errors'] - self.prev_sample['tx_errors']
            rx_errors_diff = sample['rx_errors'] - self.prev_sample['rx_errors']
            if tx_errors_diff > 0 or rx_errors_diff > 0:
                events.append(CellEvent.DATA_ERR.name)

        # ANOM_HOUR
        if sample['gps_timestamp'] is not None:
            current_hour = self._timestamp_to(sample['gps_timestamp'], 'hours')
            anomalies_in_hour = self.baselines.get('anomalies_per_hour', {}).get(current_hour, 0)
            avg_anomalies_per_hour = self.baselines.get('avg_anomalies_per_hour', 0)
            if anomalies_in_hour > avg_anomalies_per_hour:
                events.append(CellEvent.ANOM_HOUR.name)

        # ANOM_POS
        if (
            self.baselines.get('anomaly_tree') is not None
            and sample['gps_lat'] is not None
            and sample['gps_lon'] is not None
        ):
            sample_coords = np.radians([[sample['gps_lat'], sample['gps_lon']]])
            _, distances = self.baselines['anomaly_tree'].query_radius(
                sample_coords,
                r=NEIGHBOUR_RADIUS_KM / APPROX_EARTH_RADIUS_KM,
                return_distance=True,
                sort_results=True,
            )
            if distances:
                events.append(CellEvent.ANOM_POS.name)
                events.append(
                    CellEvent.ANOM_POS_INTENSITY.name + f"_{distances[0] * APPROX_EARTH_RADIUS_KM * 1000:.2f}m"
                )

        # ANOM_NAV
        if (
            'speed_bins' in self.baselines
            and 'heading_bins' in self.baselines
            and 'anomalies_nav_histogram' in self.baselines
            and sample['gps_speed'] is not None
            and sample['gps_heading'] is not None
        ):
            speed_bin = np.digitize(sample['gps_speed'], self.baselines['speed_bins'])
            heading_bin = np.digitize(sample['gps_heading'], self.baselines['heading_bins'])
            # adjust bins for zero-based indexing
            speed_bin = min(speed_bin, self.baselines['anomalies_nav_histogram'].shape[0]) - 1
            heading_bin = min(heading_bin, self.baselines['anomalies_nav_histogram'].shape[1]) - 1
            anomalies_in_bin = self.baselines['anomalies_nav_histogram'][speed_bin, heading_bin]
            avg_anomalies_nav = self.baselines.get('avg_anomalies_nav', 0)
            if anomalies_in_bin > avg_anomalies_nav:
                events.append(CellEvent.ANOM_NAV.name)

        self.prev_sample = sample

        return events

    def _process_sequence(self, sequence: List[Dict[str, Any]]) -> List[str]:
        events = []

        tx_bits = [s['tx_bits'] for s in sequence]
        tx_times = [s['gps_timestamp'] for s in sequence if s['gps_timestamp'] is not None]
        if len(tx_times) >= 2:
            timestamps = [self._timestamp_to(t, 'seconds') for t in tx_times]
            time_diffs = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
            tx_bits_diffs = [b2 - b1 for b1, b2 in zip(tx_bits[:-1], tx_bits[1:])]
            tx_bitrates = [bits / dt for bits, dt in zip(tx_bits_diffs, time_diffs) if dt > 0]

            if tx_bitrates:
                mean_bitrate = sum(tx_bitrates) / len(tx_bitrates)
                std_bitrate = (sum((br - mean_bitrate) ** 2 for br in tx_bitrates) / len(tx_bitrates)) ** 0.5
                recent_bitrate = tx_bitrates[-1]

                if recent_bitrate > mean_bitrate + 2 * std_bitrate:
                    events.append(CellEvent.DATA_BURST.name)
                if recent_bitrate < mean_bitrate - 2 * std_bitrate:
                    events.append(CellEvent.DATA_THROT.name)
        return events

    def _timestamp_to(self, timestamp_str: str, measure: str) -> int:
        dt = datetime.fromisoformat(timestamp_str)
        if measure == 'hours':
            return dt.hour
        elif measure == 'minutes':
            return dt.minute
        elif measure == 'seconds':
            return dt.second
        elif measure == 'days':
            return dt.day
        elif measure == 'months':
            return dt.month
        elif measure == 'years':
            return dt.year
        else:
            raise ValueError(f'Invalid measure: {measure}')

    def _precompute_baselines(self, data: np.ndarray, anomalies_data: np.ndarray) -> None:
        # Compute means and stds for signal features
        signal_features = ['rsrp', 'rsrq', 'sinr', 'cqi']
        for feature in signal_features:
            values = data[feature]
            self.baselines[f'{feature}_mean'] = np.nanmean(values)
            self.baselines[f'{feature}_std'] = np.nanstd(values)
            self.baselines[f'{feature}_threshold'] = self._calculate_threshold(values)

        # Compute average anomalies per hour and anomalies per hour
        anomaly_hours = anomalies_data['timestamp'].astype('datetime64[h]')
        hours, counts = np.unique(anomaly_hours, return_counts=True)
        self.baselines['anomalies_per_hour'] = dict(zip(hours.astype(str), counts))
        self.baselines['avg_anomalies_per_hour'] = np.mean(counts)

        # Build spatial index for anomalies
        if 'lat' in anomalies_data.dtype.names and 'lon' in anomalies_data.dtype.names:
            coords = np.radians(np.column_stack((anomalies_data['lat'], anomalies_data['lon'])))
            self.baselines['anomaly_tree'] = BallTree(coords, metric='haversine')
        else:
            self.baselines['anomaly_tree'] = None

        # Build anomalies navigation histogram
        speed_bins = np.arange(0, np.nanmax(data['speed']) + 10, 10)
        heading_bins = np.arange(0, 360 + 45, 45)
        self.baselines['speed_bins'] = speed_bins
        self.baselines['heading_bins'] = heading_bins
        anomalies_nav_histogram, _, _ = np.histogram2d(
            anomalies_data['speed'],
            anomalies_data['heading'],
            bins=[speed_bins, heading_bins],
        )
        self.baselines['anomalies_nav_histogram'] = anomalies_nav_histogram

        # Compute average anomalies per navigation bin
        self.baselines['avg_anomalies_nav'] = len(anomalies_data) / (len(speed_bins) * len(heading_bins))

    def _calculate_threshold(self, values: np.ndarray) -> float:
        z_scores = np.abs((values - np.nanmean(values)) / np.nanstd(values))
        threshold = np.nanmean(z_scores) + 2 * np.nanstd(z_scores)
        return threshold

    @staticmethod
    def load_combined_data(combined_csv_path: str) -> np.ndarray:
        data = np.genfromtxt(combined_csv_path, delimiter=',', names=True, dtype=None, encoding=None)
        return data

    @staticmethod
    def load_anomalies_data(anomalies_csv_paths: List[str]) -> np.ndarray:
        anomalies_list = []
        for path in anomalies_csv_paths:
            anomalies = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None)
            anomalies_list.append(anomalies)
        anomalies_data = np.concatenate(anomalies_list)
        return anomalies_data

    def _save_baselines(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.baselines, f)
        except Exception as e:
            raise Exception(f'Error saving baselines: {e}')

    def _load_baselines(self, filename: str):
        try:
            with open(filename, 'rb') as f:
                self.baselines = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'Baselines file {filename} not found')
        except Exception as e:
            raise Exception(f'Error loading baselines: {e}')

    @classmethod
    def from_csv(
        cls,
        combined_csv_path: str,
        anomalies_csv_paths: List[str],
        baselines_output_path: str,
    ) -> Self:
        instance = cls(None)
        data = instance.load_combined_data(combined_csv_path)
        anomalies_data = instance.load_anomalies_data(anomalies_csv_paths)
        instance._precompute_baselines(data, anomalies_data)
        instance._save_baselines(baselines_output_path)
        return instance


if __name__ == '__main__':
    # Sample usage
    baselines_file_path = '/home/gstwebrtcapp/tools/cellular-controller/objects/event_classifier/baselines.pkl'
    # classifier = CellEventClassifier.from_csv(
    #     '/home/gstwebrtcapp/tools/cellular-controller/combined.csv',
    #     ['/home/gstwebrtcapp/tools/cellular-controller/anomalies_combined.csv'],
    #     baselines_file_path,
    # )

    classifier = CellEventClassifier(baselines_file_path)

    test_report = CellReport(
        identity='5G-D2-WAVELAB',
        phy_cell_id=1,
        data_class='5G SA',
        cqi=10,
        ri=1,
        mcs=10,
        modulation='qpsk',
        rsrp=-100,
        rsrq=-10,
        sinr=10,
        rx_packets=10,
        rx_bits=1000,
        rx_drops=0,
        rx_errors=0,
        tx_errors=0,
        tx_bits=1000,
        tx_packets=10,
        tx_drops=1,
        tx_queue_drops=0,
        gps_speed=50,
        gps_heading=180,
        gps_lat=40.7128,
        gps_lon=54.0060,
        gps_timestamp='2024-08-01 12:30:00',
    )

    print(classifier.identify_events(test_report))

    # sample = {
    #     'phy_cell_id': 1,
    #     'network_type': 1,
    #     'rsrp': -100,
    #     'rsrq': -10,
    #     'sinr': 10,
    #     'cqi': 10,
    #     'tx_errors': 0,
    #     'rx_errors': 0,
    #     'heading': 180,
    #     'tx_bits': 1000,
    #     'tx_packets': 10,
    #     'gps_speed': 50,
    #     'gps_heading': 180,
    #     'gps_lat': 40.7128,
    #     'gps_lon': -74.0060,
    #     'gps_timestamp': '2021-08-01 00:00:00',
    # }
    # point_events, sequence_events = classifier.process_sample(sample)
