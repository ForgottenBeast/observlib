from .prometheus_exporter import PrometheusClientExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics import AlwaysOnExemplarFilter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry import metrics
from .legacy_prometheus_metrics import start_server
from observlib.decorator import set_exec_time_histogram
from prometheus_client import REGISTRY, CollectorRegistry

_fallback_registry: CollectorRegistry = REGISTRY
_service_name: str = "default"
_prometheus_metrics = []  # keep references alive to prevent GC


def get_prometheus_client_exporter(registry=None):
    registry = registry or REGISTRY
    exporter = PrometheusClientExporter(registry=registry)
    return PeriodicExportingMetricReader(exporter)


def otel_configured():
    provider = metrics.get_meter_provider()
    return isinstance(provider, MeterProvider)


def configure_metrics(
    legacy_prometheus_config, server, service_name, resource, prometheus_registry=None
):
    global metric_reader
    global meter
    legacy_prometheus_port = int(legacy_prometheus_config.split(":")[1])

    if legacy_prometheus_port == 0 and metric_reader:
        metrics_readers = [
            metric_reader,
            PrometheusClientExporter(prometheus_registry),
        ]
    elif legacy_prometheus_port != 0:
        metrics_readers = [
            PrometheusMetricReader(),
            PrometheusClientExporter(prometheus_registry),
        ]
    else:
        metrics_readers = [
            metric_reader,
            PrometheusMetricReader(),
            PrometheusClientExporter(prometheus_registry),
        ]

    if legacy_prometheus_port != 0 or server:
        provider = MeterProvider(
            metric_readers=metrics_readers,
            exemplar_filter=AlwaysOnExemplarFilter(),
            resource=resource,
        )

        # Sets the global default meter provider
        metrics.set_meter_provider(provider)

    meter = metrics.get_meter(service_name)
    set_exec_time_histogram(
        meter.create_histogram(
            name="function_exec_time_seconds",
            description="Execution time of wrapped functions",
            unit="s",
        )
    )

    if legacy_prometheus_port != 0:
        start_server(legacy_prometheus_config)
