from opentelemetry.trace import Link, SpanContext, TraceFlags
from opentelemetry.sdk.trace import Status, StatusCode
from opentelemetry import trace

sname = None
def set_sname(name):
    global sname
    sname = sname

def span_from_context(span_name, trace_id, span_id):
    global sname
    parent_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return trace.get_tracer(sname).start_as_current_span(
        span_name, links=[Link(parent_context)]
    )


def set_span_error_status():
    current_span = trace.get_current_span()
    current_span.set_status(Status(StatusCode.ERROR))
