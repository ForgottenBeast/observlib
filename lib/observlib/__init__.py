from functools import wraps
import logging
from pyroscope.otel import PyroscopeSpanProcessor
from opentelemetry import trace, metrics
from opentelemetry.trace import Link, SpanContext, TraceFlags, INVALID_SPAN_CONTEXT

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

# Creates a meter from the global meter provider
meter = None

def traced(span_name: str = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = span_name or func.__name__
            with trace.get_tracer().start_as_current_span(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def span_from_context(span_name,trace_id, span_id):
    parent_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED)
    )
    return tracer.start_as_current_span(span_name, links = [Link(parent_context)])

def set_span_error_status():
    current_span = trace.get_current_span()
    current_span.set_status(Status(StatusCode.ERROR))


def get_meter():
    global meter
    return meter


def get_tracer():
    return trace.get_tracer(__name__)


def get_trace():
    global trace
    return trace

def configure_telemetry(service_name, server, pyroscope_server, devMode = False):
    global meter
    if devMode:
        sample_rate = 100
    else:
        sample_rate = 5

    pyroscope.configure(
        application_name=service_name,
        server_address="http://{}".format(pyroscope_server),
        sample_rate = sample_rate
    )

    endpoint = "http://{}/v1/traces".format(server)
    resource = Resource.create(attributes={"service.name": service_name})


    tracerProvider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    tracerProvider.add_span_processor(PyroscopeSpanProcessor())
    tracerProvider.add_span_processor(processor)
    trace.set_tracer_provider(tracerProvider)

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
