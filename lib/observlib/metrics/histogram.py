from typing import Optional, List
from observlib.metrics import otel_configured, _prometheus_metrics, _fallback_registry
from prometheus_client import Histogram as PromHistogram
from opentelemetry import metrics


class Histogram:
    def __init__(
        self,
        name: str,
        description: str = "",
        labelnames: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ):
        self._labels = labelnames or []
        self._is_otel = otel_configured()

        if self._is_otel:
            self._instrument = metrics.get_meter().create_histogram(
                name, description=description
            )
            self._children = {}
        else:
            kwargs = {"registry": _fallback_registry}
            if buckets is not None:
                kwargs["buckets"] = buckets
            self._instrument = PromHistogram(name, description, self._labels, **kwargs)
            _prometheus_metrics.append(self._instrument)  # prevent GC

    def labels(self, **labels) -> "Histogram":
        if not self._labels:
            raise ValueError("No labels defined for this metric")
        if self._is_otel:
            key = tuple(sorted(labels.items()))
            if key not in self._children:
                self._children[key] = self._instrument.bind(labels)
            return self._children[key]
        else:
            return self._instrument.labels(**labels)

    def observe(self, amount: float):
        if self._is_otel:
            self._instrument.record(amount)
        else:
            self._instrument.observe(amount)


def create_histogram(
    name: str,
    description: str = "",
    labelnames: Optional[List[str]] = None,
    buckets: Optional[List[float]] = None,
) -> Histogram:
    return Histogram(name, description, labelnames, buckets)
