from typing import Optional, List
from opentelemetry import metrics
from prometheus_client import Gauge as PromGauge
from observlib.metrics import otel_configured, _prometheus_metrics, _fallback_registry


class Gauge:
    def __init__(
        self,
        name: str,
        description: str = "",
        labelnames: Optional[List[str]] = None,
        unit="1",
    ):
        self._labels = labelnames or []
        self._is_otel = otel_configured()

        if self._is_otel:
            self._instrument = metrics.get_meter().create_gauge(
                name=name, unit=unit, description=description
            )
            self.children = {}
        else:
            self._instrument = PromGauge(
                name, description, self._labels, registry=_fallback_registry
            )
            _prometheus_metrics.append(self._instrument)  # prevent GC

    def labels(self, **labels) -> "Gauge":
        if not self._labels:
            raise ValueError("No labels defined for this metric")

        if self._is_otel:
            # OTel API returns bound instruments for labels
            key = tuple(sorted(labels.items()))
            if key not in self._children:
                self._children[key] = self._instrument.bind(labels)
            return self._children[key]
        else:
            return self._instrument.labels(**labels)

    def set(self, value: float):
        self._instrument.set(value)


def create_gauge(
    name: str, description: str = "", labelnames: Optional[List[str]] = None, unit="1"
) -> Gauge:
    return Gauge(name, description, labelnames, unit)
