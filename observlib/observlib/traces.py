import logging

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pyroscope.otel import PyroscopeSpanProcessor
from opentelemetry import trace

logger = logging.getLogger(__name__)


def configure_tracing(server, resource):
    endpoint = f"http://{server}/v1/traces"

    try:
        tracerProvider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        tracerProvider.add_span_processor(PyroscopeSpanProcessor())
        tracerProvider.add_span_processor(processor)
        trace.set_tracer_provider(tracerProvider)
    except Exception as e:
        logger.error(f"Failed to configure tracing with endpoint {endpoint}: {e}")
        raise
