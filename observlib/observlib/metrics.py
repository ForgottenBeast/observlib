import logging

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry import metrics
from opentelemetry.sdk.metrics import AlwaysOnExemplarFilter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

logger = logging.getLogger(__name__)


def configure_metrics(server, resource):
    # always make metrics available to someone running a prometheus server
    service_name = resource.attributes.get("service.name", "unknown")
    metric_readers = [PrometheusMetricReader(service_name)]

    if server:
        try:
            # if someone runs an otlp exporter, the metrics will be sent there too
            otlp_exporter = OTLPMetricExporter(
                endpoint=f"http://{server}/v1/metrics"
            )
            metric_readers.append(
                PeriodicExportingMetricReader(otlp_exporter, export_interval_millis=5000)
            )
        except Exception as e:
            logger.error(f"Failed to configure OTLP metrics exporter: {e}")
            raise

    try:
        provider = MeterProvider(
            metric_readers=metric_readers,
            exemplar_filter=AlwaysOnExemplarFilter(),
            resource=resource,
        )

        # Sets the global default meter provider
        metrics.set_meter_provider(provider)
    except Exception as e:
        logger.error(f"Failed to configure meter provider: {e}")
        raise
