from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics import AlwaysOnExemplarFilter
from opentelemetry import metrics
from .legacy_prometheus_metrics import start_server
from observlib.decorator import set_exec_time_histogram

meter = None
metric_reader = None


def get_meter():
    global meter
    return meter


def set_metric_reader(reader):
    global metric_reader
    metric_reader = reader


def configure_metrics(legacy_prometheus_config, server, service_name, resource):
    global metric_reader
    global meter
    legacy_prometheus_port = int(legacy_prometheus_config.split(":")[1])

    if legacy_prometheus_port == 0 and metric_reader:
        metrics_readers = [metric_reader]
    elif legacy_prometheus_port != 0:
        metrics_readers = [
            PrometheusMetricReader(),
        ]
    else:
        metrics_readers = [
            metric_reader,
            PrometheusMetricReader(),
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
