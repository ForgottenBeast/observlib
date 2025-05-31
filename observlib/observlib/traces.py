from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pyroscope.otel import PyroscopeSpanProcessor
from opentelemetry import trace


def configure_tracing(server, resource):
    endpoint = "http://{}/v1/traces".format(server)

    tracerProvider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    tracerProvider.add_span_processor(PyroscopeSpanProcessor())
    tracerProvider.add_span_processor(processor)
    trace.set_tracer_provider(tracerProvider)
