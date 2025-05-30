from opentelemetry import metrics
from prometheus_client import REGISTRY
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

def (service_name, registry = None);
    global meter
    meter = metrics.get_meter(service_name)
    registry = registry or REGISTRY

    exporter = PrometheusClientExporter(registry=registry)
    return PeriodicExportingMetricReader(exporter)
