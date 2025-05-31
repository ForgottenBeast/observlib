from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider


def otel_configured():
    provider = metrics.get_meter_provider()
    return isinstance(provider, MeterProvider)
