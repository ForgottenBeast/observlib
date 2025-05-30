from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pyroscope.otel import PyroscopeSpanProcessor
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from .metrics import set_metric_reader


def get_trace():
    global trace
    return trace


def configure_tracing(server, resource):
    endpoint = "http://{}/v1/traces".format(server)

    tracerProvider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    tracerProvider.add_span_processor(PyroscopeSpanProcessor())
    tracerProvider.add_span_processor(processor)
    trace.set_tracer_provider(tracerProvider)

    otlp_exporter = OTLPMetricExporter(endpoint="http://{}/v1/metrics".format(server))
    set_metric_reader(
        PeriodicExportingMetricReader(otlp_exporter, export_interval_millis=5000)
    )
