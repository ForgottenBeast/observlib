import pyroscope
from prometheus_client import CollectorRegistry, push_to_gateway
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace.span import format_trace_id
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

class Observer:
    def __init__(self, prometheus_gtw = None, pyroscope_server = None, otlp_server = None, prometheus_port = None, service_name)
        self.prometheus_gtw = prometheus_gtw
        self.pyroscope_server = pyroscope_server
        self.otlp_server = otlp_server

        if otlp_server:
            res = Resource(attributes = {
            SERVICE_NAME: service_name
            })
            tracerProvider = TracerProvider(resource=resource)
            processor = BatchSpanProcessor(OTLPSpanExporter(endpoint = otlp_server))
            tracerProvider.add_span_processor(processor)
            trace.set_tracer_provider(tracerProvider)
            self.tracer = trace.get_tracer(__name__)

        if pyroscope_server:
            pyroscope.configure(application_name = service_name,
                                server_address = pyroscope_server)


        self.registry = CollectorRegistry()
        self.service_name = service_name

    def push_metrics(self):
        push_to_gateway(self.prometheus_gtw, job = self.service_name, registry = self.registry)

