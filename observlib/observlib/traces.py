import logging
from beartype import beartype
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pyroscope.otel import PyroscopeSpanProcessor
from opentelemetry import trace

logger: logging.Logger = logging.getLogger(__name__)


@beartype
def configure_tracing(server: str, resource: Resource) -> None:
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
