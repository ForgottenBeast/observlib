from opentelemetry.sdk.metrics.export import MetricExporter, AggregationTemporality
from prometheus_client import CollectorRegistry, Gauge, Counter
from collections import defaultdict

class PrometheusClientExporter(MetricExporter):
    def __init__(self, registry=None):
        self.registry = registry or CollectorRegistry()
        self.metrics_map = {}  # maps OTLP metric names to Prometheus metrics

    def export(self, metrics):
        for record in metrics:
            name = record.name
            description = record.description
            labels = dict(record.attributes)

            # Choose type based on OTLP instrument
            if record.instrument.type.name == "COUNTER":
                if name not in self.metrics_map:
                    self.metrics_map[name] = Counter(name, description, labels.keys(), registry=self.registry)
                self.metrics_map[name].labels(**labels).inc(record.point.value)

            elif record.instrument.type.name == "OBSERVABLE_GAUGE":
                if name not in self.metrics_map:
                    self.metrics_map[name] = Gauge(name, description, labels.keys(), registry=self.registry)
                self.metrics_map[name].labels(**labels).set(record.point.value)

        return self._success_result()

    def force_flush(self, timeout_millis: int = 10000):
        return True

    def shutdown(self, timeout_millis: int = 30000):
        return True

    def aggregation_temporality(self, instrument_type):
        return AggregationTemporality.CUMULATIVE
