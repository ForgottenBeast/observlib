from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry import metrics
from opentelemetry.sdk.metrics import AlwaysOnExemplarFilter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader


def configure_metrics(server, resource):
    # always make metrics available to someone running a prometheus server
    metric_readers = [PrometheusMetricReader(resource.attributes["service.name"])]

    if server:
        # if someone runs an otlp exporter, the metrics will be sent there too
        otlp_exporter = OTLPMetricExporter(
            endpoint="http://{}/v1/metrics".format(server)
        )
        metric_readers.append(
            PeriodicExportingMetricReader(otlp_exporter, export_interval_millis=5000)
        )

    provider = MeterProvider(
        metric_readers=metric_readers,
        exemplar_filter=AlwaysOnExemplarFilter(),
        resource=resource,
    )

    # Sets the global default meter provider
    metrics.set_meter_provider(provider)
