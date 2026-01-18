import logging
from typing import Optional, Any
from opentelemetry.sdk.resources import Resource

import pyroscope
from .decorator import traced as traced

from .traces import configure_tracing
from .logs import configure_logging
from .metrics import configure_metrics


def configure_telemetry(
    service_name: str,
    server: Optional[str] = None,
    pyroscope_server: Optional[str] = None,
    pyroscope_sample_rate: int = 5,
    log_level: int = logging.INFO,
    resource_attrs: Optional[dict[str, Any]] = None,
) -> None:
    if not service_name or not isinstance(service_name, str):
        raise ValueError("service_name must be a non-empty string")

    if resource_attrs is None:
        resource_attrs = {}

    if pyroscope_server:
        pyroscope.configure(
            application_name=service_name,
            server_address=f"http://{pyroscope_server}",
            sample_rate=pyroscope_sample_rate,
        )

    resource = Resource.create(
        attributes={"service.name": service_name} | resource_attrs
    )

    if server:
        configure_tracing(server, resource)

    configure_logging(server, resource, log_level)
    configure_metrics(server, resource)
