import logging
from pyroscope.otel import PyroscopeSpanProcessor
from opentelemetry import trace, metrics

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.urllib import URLLibInstrumentor

from opentelemetry._logs import set_logger_provider

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, Status, StatusCode
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)
from opentelemetry.sdk.metrics import AlwaysOnExemplarFilter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

import pyroscope


def strip_query_params(url: str) -> str:
    return url.split("?")[0]


URLLibInstrumentor().instrument(
    # Remove all query params from the URL attribute on the span.
    url_filter=strip_query_params,
)
AsyncioInstrumentor().instrument()

# Creates a tracer from the global tracer provider
tracer = None


# Creates a meter from the global meter provider
meter = None

def set_span_error_status():
    current_span = trace.get_current_span()
    current_span.set_status(Status(StatusCode.ERROR))



def get_meter():
    global meter
    return meter


def get_tracer():
    global tracer
    return tracer

def get_trace():
    global trace
    return trace


def configure_telemetry(server, service_name, pyroscope_server):
    global tracer
    global meter
    pyroscope.configure(
        application_name=service_name,
        server_address="http://{}".format(pyroscope_server),
    )

    endpoint = "http://{}/v1/traces".format(server)
    resource = Resource.create(attributes={"service.name": service_name})

    tracerProvider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    tracerProvider.add_span_processor(PyroscopeSpanProcessor())
    tracerProvider.add_span_processor(processor)
    trace.set_tracer_provider(tracerProvider)
    tracer = trace.get_tracer(__name__)

    otlp_exporter = OTLPMetricExporter(endpoint="http://{}/v1/metrics".format(server))
    metric_reader = PeriodicExportingMetricReader(
        otlp_exporter, export_interval_millis=5000
    )

    provider = MeterProvider(
        metric_readers=[metric_reader],
        exemplar_filter=AlwaysOnExemplarFilter(),
        resource=resource,
    )

    # Sets the global default meter provider
    metrics.set_meter_provider(provider)

    meter = metrics.get_meter(service_name)

    otlp_log_exporter = OTLPLogExporter(endpoint="http://{}/v1/logs".format(server))

    # Set up the logger provider with a batch log processor
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(otlp_log_exporter))
    set_logger_provider(logger_provider)

    # Set up Python logging integration
    handler = LoggingHandler(level=logging.DEBUG, logger_provider=logger_provider)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
