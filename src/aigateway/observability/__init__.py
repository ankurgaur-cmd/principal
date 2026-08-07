from .alerts import Alert, AlertCentre, no_models_available
from .baselines import LatencyBaselines, segment_key
from .db import SqliteRecordSink
from .fleet import FleetStats
from .hops import Hop, TraceContext
from .record import RecordSink, RequestRecord


def build_sink(path: str, fleet=None):
    """SQLite unless the configured path says JSONL.

    The extension is the switch: it keeps the legacy sink reachable without a
    second setting that could disagree with the path it describes.
    """
    if str(path).endswith(".jsonl"):
        return RecordSink(path, fleet=fleet)
    return SqliteRecordSink(path, fleet=fleet)


__all__ = [
    "build_sink",
    "SqliteRecordSink",
    "Alert",
    "AlertCentre",
    "FleetStats",
    "Hop",
    "LatencyBaselines",
    "TraceContext",
    "RecordSink",
    "RequestRecord",
    "no_models_available",
    "segment_key",
]
