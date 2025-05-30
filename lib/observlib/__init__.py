import .autoinstrumentation
from .legacy_prometheus_metrics import start_server
import time
from functools import wraps
import asyncio
import logging
from pyroscope.otel import PyroscopeSpanProcessor
from opentelemetry import trace, metrics
from opentelemetry.trace import Link, SpanContext, TraceFlags

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader


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



sname = None

exec_time_histogram = None


# Creates a meter from the global meter provider
meter = None


def traced(func):
    global sname

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        with trace.get_tracer(sname).start_as_current_span(func.__name__):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                exec_time_histogram.record(
                    time.perf_counter() - start, {"function": func.__name__}
                )

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        with trace.get_tracer(sname).start_as_current_span(func.__name__):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                exec_time_histogram.record(
                    time.perf_counter() - start, {"function": func.__name__}
                )

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def span_from_context(span_name, trace_id, span_id):
    parent_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return trace.get_tracer().start_as_current_span(
        span_name, links=[Link(parent_context)]
    )


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


def configure_telemetry(
    service_name,
    server=None,
    pyroscope_server=None,
    devMode=False,
    legacy_prometheus_config="127.0.0.1:0",
):
    legacy_prometheus_port = int(legacy_prometheus_config.split(":")[1])
    global sname
    sname = service_name
    global meter
    global exec_time_histogram
    if devMode:
        sample_rate = 100
    else:
        sample_rate = 5

    if pyroscope_server:
        pyroscope.configure(
            application_name=service_name,
            server_address="http://{}".format(pyroscope_server),
            sample_rate=sample_rate,
        )

    metric_reader = None
    resource = Resource.create(attributes={"service.name": service_name})

    if server:
        endpoint = "http://{}/v1/traces".format(server)

        tracerProvider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        tracerProvider.add_span_processor(PyroscopeSpanProcessor())
        tracerProvider.add_span_processor(processor)
        trace.set_tracer_provider(tracerProvider)

        otlp_exporter = OTLPMetricExporter(
            endpoint="http://{}/v1/metrics".format(server)
        )
        metric_reader = PeriodicExportingMetricReader(
            otlp_exporter, export_interval_millis=5000
        )

    if legacy_prometheus_port == 0 and metric_reader:
        metrics_readers = [metric_reader]
    elif legacy_prometheus_port != 0:
        metrics_readers = [
            PrometheusMetricReader(),
        ]
    else:
        metrics_readers = [
            metric_reader,
            PrometheusMetricReader(),
        ]

    if legacy_prometheus_port != 0 or server:
        provider = MeterProvider(
            metric_readers=metrics_readers,
            exemplar_filter=AlwaysOnExemplarFilter(),
            resource=resource,
        )

        # Sets the global default meter provider
        metrics.set_meter_provider(provider)

        meter = metrics.get_meter(service_name)
        exec_time_histogram = meter.create_histogram(
            name="function_exec_time_seconds",
            description="Execution time of wrapped functions",
            unit="s",
        )

    if server:
        otlp_log_exporter = OTLPLogExporter(endpoint="http://{}/v1/logs".format(server))

        # Set up the logger provider with a batch log processor
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(otlp_log_exporter)
        )
        set_logger_provider(logger_provider)

        # Set up Python logging integration
        handler = LoggingHandler(level=logging.DEBUG, logger_provider=logger_provider)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.DEBUG)

    if legacy_prometheus_port != 0:
        start_server(legacy_prometheus_config)
