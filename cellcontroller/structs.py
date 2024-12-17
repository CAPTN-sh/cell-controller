from dataclasses import dataclass
from typing import Any, Dict, List, Self, Type

from cellcontroller.common.utils import LOGGER


@dataclass
class CellReport:
    identity: str
    phy_cell_id: int
    data_class: str
    cqi: int
    ri: int
    mcs: int
    modulation: str
    rsrp: int
    rsrq: int
    sinr: int
    rx_packets: int
    rx_bits: int
    rx_drops: int
    rx_errors: int
    tx_packets: int
    tx_bits: int
    tx_drops: int
    tx_queue_drops: int
    tx_errors: int
    gps_lat: float | None = None
    gps_lon: float | None = None
    gps_speed: float | None = None
    gps_heading: float | None = None
    gps_timestamp: str | None = None


@dataclass
class CellPayload:
    identity: str

    lat: str
    lon: str
    altitude: str
    speed: str
    heading: str
    date: str
    time: str

    operator: str
    cell_id: str
    enb_id: str
    sector_id: str
    phy_cell_id: str
    data_class: str
    primary_band: str
    lca_band: List[str]
    lulca_band: List[str]

    l_dl_modulation: str
    l_cqi: str
    l_ri: str
    l_mcs: str
    l_rssi: str
    l_rsrp: str
    l_rsrq: str
    l_sinr: str

    nr_dl_modulation: str
    nr_rsrp: str
    nr_rsrq: str
    nr_sinr: str

    rx_packets: int
    rx_bits: int
    rx_drops: int
    rx_errors: int
    tx_packets: int
    tx_bits: int
    tx_drops: int
    tx_queue_drops: int
    tx_errors: int

    @classmethod
    def from_mqtt_payload(cls, payload: Dict[str, Any]) -> Self:
        return cls(
            identity=payload.get("identity", None),
            lat=payload.get("gps", {}).get("lat", None),
            lon=payload.get("gps", {}).get("lon", None),
            altitude=payload.get("gps", {}).get("altitude", None),
            speed=payload.get("gps", {}).get("speed", None),
            heading=payload.get("gps", {}).get("heading", None),
            date=payload.get("date", None),
            time=payload.get("time", None),
            operator=CellPayload.map_operator(payload.get("lte", {}).get("lCurrentOperator", None)),
            cell_id=payload.get("lte", {}).get("lCurrentCellid", None),
            enb_id=payload.get("lte", {}).get("lEnbId", None),
            sector_id=payload.get("lte", {}).get("lSectorId", None),
            phy_cell_id=payload.get("lte", {}).get("lPhyCellId", None),
            data_class=payload.get("lte", {}).get("lDataClass", None),
            primary_band=payload.get("lte", {}).get("lPrimaryBand", None),
            lca_band=payload.get("lte", {}).get("lcaBand", []),
            lulca_band=payload.get("lte", {}).get("lulcaBand", []),
            l_dl_modulation=payload.get("lte", {}).get("lDlModulation", None),
            l_cqi=payload.get("lte", {}).get("lCqi", None),
            l_ri=payload.get("lte", {}).get("lRi", None),
            l_mcs=payload.get("lte", {}).get("lMcs", None),
            l_rssi=payload.get("lte", {}).get("lRssi", None),
            l_rsrp=payload.get("lte", {}).get("lRsrp", None),
            l_rsrq=payload.get("lte", {}).get("lRsrq", None),
            l_sinr=payload.get("lte", {}).get("lSinr", None),
            nr_dl_modulation=payload.get("lte", {}).get("lNrDlModulation", None),
            nr_rsrp=payload.get("lte", {}).get("lNrRsrp", None),
            nr_rsrq=payload.get("lte", {}).get("lNrRsrq", None),
            nr_sinr=payload.get("lte", {}).get("lNrSinr", None),
            rx_packets=int(payload.get("lte", {}).get("lrxpacketspersecond", 0)),
            rx_bits=int(payload.get("lte", {}).get("lrxbitspersecond", 0)),
            rx_drops=int(payload.get("lte", {}).get("lrxdropspersecond", 0)),
            rx_errors=int(payload.get("lte", {}).get("lrxerrorspersecond", 0)),
            tx_packets=int(payload.get("lte", {}).get("ltxpacketspersecond", 0)),
            tx_bits=int(payload.get("lte", {}).get("ltxbitspersecond", 0)),
            tx_drops=int(payload.get("lte", {}).get("ltxdropspersecond", 0)),
            tx_queue_drops=int(payload.get("lte", {}).get("ltxqueuedropspersecond", 0)),
            tx_errors=int(payload.get("lte", {}).get("ltxerrorspersecond", 0)),
        )

    def to_report(self) -> CellReport:
        prefix = "nr" if self.data_class == "5G NSA" else "l"
        return CellReport(
            identity=CellPayload.cast_fields(self.identity.upper() if self.identity else None, str),
            phy_cell_id=CellPayload.cast_fields(self.phy_cell_id, int),
            data_class=CellPayload.cast_fields(self.data_class.upper() if self.data_class else None, str),
            cqi=CellPayload.cast_fields(self.l_cqi, int),
            ri=CellPayload.cast_fields(self.l_ri, int),
            mcs=CellPayload.cast_fields(self.l_mcs, int),
            modulation=CellPayload.cast_fields(self._get_attr_by_prefix(prefix, "dl_modulation"), str),
            rsrp=CellPayload.cast_fields(self._get_attr_by_prefix(prefix, "rsrp"), int),
            rsrq=CellPayload.cast_fields(self._get_attr_by_prefix(prefix, "rsrq"), int),
            sinr=CellPayload.cast_fields(self._get_attr_by_prefix(prefix, "sinr"), int),
            rx_packets=self.rx_packets,
            rx_bits=self.rx_bits,
            rx_drops=self.rx_drops,
            rx_errors=self.rx_errors,
            tx_packets=self.tx_packets,
            tx_bits=self.tx_bits,
            tx_drops=self.tx_drops,
            tx_queue_drops=self.tx_queue_drops,
            tx_errors=self.tx_errors,
            gps_lat=CellPayload.cast_fields(self.lat, float),
            gps_lon=CellPayload.cast_fields(self.lon, float),
            gps_speed=CellPayload.cast_fields(CellPayload.trim_units(self.speed), float),
            gps_heading=CellPayload.cast_fields(CellPayload.trim_units(self.heading), float),
            gps_timestamp=f"{self.date} {self.time}" if self.date and self.time else None,
        )

    def _get_attr_by_prefix(self, prefix: str, attr: str) -> Any | None:
        return getattr(self, f"{prefix}_{attr}", None)

    @staticmethod
    def map_operator(operator: str) -> str:
        if operator.lower().startswith("vodafone"):
            return "vodafone"
        elif operator.lower().startswith("telekom"):
            return "telekom"
        else:
            LOGGER.warning(f"CellPayload: operator {operator} is not recognized")
            return ""

    @staticmethod
    def cast_fields(val: Any, cast_to: Type | None = None, fallback: Type = float) -> Any | None:
        if val:
            if cast_to is not None:
                try:
                    val = cast_to(val)
                    return val
                except ValueError:
                    try:
                        f = cast_to(fallback(val))
                        return f
                    except ValueError:
                        return None
            else:
                return val
        return None

    @staticmethod
    def trim_units(val: str | None) -> str | None:
        return val.split(" ")[0] if val else None
