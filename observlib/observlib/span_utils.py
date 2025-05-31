from opentelemetry.sdk.trace import Status, StatusCode
from opentelemetry import trace


def set_span_error_status():
    current_span = trace.get_current_span()
    current_span.set_status(Status(StatusCode.ERROR))
