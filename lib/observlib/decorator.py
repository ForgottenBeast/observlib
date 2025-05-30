from functools import wraps
from opentelemetry import trace
import time
import asyncio

sname = None
exec_time_histogram = None


def set_exec_time_histogram(histogram):
    global exec_time_histogram
    exec_time_histogram = histogram


def set_sname(name):
    global sname
    sname = sname


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
