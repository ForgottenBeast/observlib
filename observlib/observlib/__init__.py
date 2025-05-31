import logging
from opentelemetry.sdk.resources import Resource

import pyroscope
from .decorator import traced as traced
from .span_utils import (
    span_from_context as span_from_context,
    set_span_error_status as set_span_error_status,
)

from .traces import configure_tracing
from .logs import configure_logging
from .metrics import configure_metrics
from .globals import set_sname


def configure_telemetry(
    service_name,
    server=None,
    pyroscope_server=None,
    devMode=False,
    configure_provider=False,
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
        if configure_provider:
            configure_tracing(server, resource)

        if devMode:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        configure_logging(server, resource, log_level, configure_provider)

    configure_metrics(server, resource, configure_provider)
