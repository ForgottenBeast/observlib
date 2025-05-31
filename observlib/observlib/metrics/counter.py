from typing import Optional, List
from observlib.metrics import otel_configured, _prometheus_metrics, _fallback_registry
from opentelemetry import metrics
from prometheus_client import Counter as PromCounter


class Counter:
    def __init__(
        self,
        name: str,
        description: str = "",
        labelnames: Optional[List[str]] = None,
        unit: str = "1",
    ):
        self._labels = labelnames or []
        self._is_otel = otel_configured()

        if self._is_otel:
            self._instrument = metrics.get_meter().create_counter(
                name, description=description, unit=unit
            )
            self._children = {}
        else:
            self._instrument = PromCounter(
                name, description, self._labels, registry=_fallback_registry
            )
            _prometheus_metrics.append(self._instrument)  # prevent GC

    def labels(self, **labels) -> "Counter":
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

    def inc(self, amount: float = 1.0):
        if self._is_otel:
            self._instrument.add(amount)
        else:
            self._instrument.inc(amount)


def create_counter(name, description, unit, labelnames) -> Counter:
    return Counter(name, description, unit=unit, labelnames=labelnames)
