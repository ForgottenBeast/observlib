import logging
from opentelemetry.sdk.resources import Resource

import pyroscope
from .decorator import set_sname, traced as traced
from .span_utils import (
    span_from_context as span_from_context,
    set_span_error_status as set_span_error_status,
)

from .traces import configure_tracing, get_trace as get_trace
from .logs import configure_logging
from .metrics import configure_metrics, get_meter as get_meter


def configure_telemetry(
    service_name,
    server=None,
    pyroscope_server=None,
    devMode=False,
    legacy_prometheus_config="127.0.0.1:0",
):
    set_sname(service_name)
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

    resource = Resource.create(attributes={"service.name": service_name})

    if server:
        configure_tracing(server, resource, service_name)

        if devMode:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        configure_logging(server, resource, log_level, service_name)

    configure_metrics(legacy_prometheus_config, server, service_name, resource)
