import pyroscope
from prometheus_client import CollectorRegistry, push_to_gateway
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace.span import format_trace_id
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# global observer object
OBSERVER = None


class Observer:
    def __init__(
        self,
        service_name,
        prometheus_gtw=None,
        pyroscope_server=None,
        otlp_server=None,
    ):
        self.prometheus_gtw = prometheus_gtw
        self.pyroscope_server = pyroscope_server

        if otlp_server:
            self.otlp_server = otlp_server + "/v1/traces"

        if otlp_server:
            resource = Resource(attributes={SERVICE_NAME: service_name})
            tracerProvider = TracerProvider(resource=resource)
            processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=self.otlp_server))
            tracerProvider.add_span_processor(processor)
            trace.set_tracer_provider(tracerProvider)
            self.tracer = trace.get_tracer(__name__)

        if pyroscope_server:
            pyroscope.configure(
                application_name=service_name, server_address=pyroscope_server
            )

        self.registry = CollectorRegistry()
        self.service_name = service_name

    def push_metrics(self):
        push_to_gateway(
            self.prometheus_gtw, job=self.service_name, registry=self.registry
        )


def get_trace_id(span):
    return format_trace_id(span.get_span_context().trace_id)
