import logging
from opentelemetry.sdk.resources import Resource

import pyroscope
from .decorator import traced as traced

from .traces import configure_tracing
from .logs import configure_logging
from .metrics import configure_metrics


def configure_telemetry(
    service_name,
    server=None,
    pyroscope_server=None,
    devMode=False,
):
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
        configure_tracing(server, resource)

        if devMode:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        configure_logging(server, resource, log_level)

    configure_metrics(server, resource)
