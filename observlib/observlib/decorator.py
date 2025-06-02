from functools import wraps
from opentelemetry import trace
import time
import asyncio


def traced(
    timing_histogram=None,
    success_counter=None,
    failure_counter=None,
    counter_factory=None,
    timer_factory=None,
    label_fn=None,
    amount_fn=None,
    tracer=None,
):
    def decorator(func):
        def resolve(maybe_callable, args=[]):
            return maybe_callable(*args) if callable(maybe_callable) else maybe_callable

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            with trace.get_tracer(tracer).start_as_current_span(func.__name__):
                result = None
                error = None
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as ex:
                    error = ex
                    raise

                finally:
                    if timing_histogram:
                        exec_time_histogram = resolve(timer_factory, [timing_histogram])
                        exec_time_histogram.record(
                            time.perf_counter() - start, {"function": func.__name__}
                        )

                    labels = (
                        label_fn(result=result, exception=error) if label_fn else {}
                    )
                    amount = (
                        amount_fn(result=result, exception=error) if amount_fn else 1
                    )
                    counter_name = failure_counter if error else success_counter
                    counter = resolve(counter_factory, [counter_name])
                    if counter:
                        counter.add(amount, attributes=labels)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            with trace.get_tracer(tracer).start_as_current_span(func.__name__):
                result = None
                error = None
                try:
                    result = await func(*args, **kwargs)
                    return result

                except Exception as ex:
                    error = ex
                    raise

                finally:
                    if timing_histogram:
                        exec_time_histogram = resolve(timer_factory, [timing_histogram])
                        exec_time_histogram.record(
                            time.perf_counter() - start, {"function": func.__name__}
                        )

                    labels = (
                        label_fn(result=result, exception=error) if label_fn else {}
                    )
                    amount = (
                        amount_fn(result=result, exception=error) if amount_fn else 1
                    )
                    counter_name = failure_counter if error else success_counter
                    counter = resolve(counter_factory, [counter_name])
                    if counter:
                        counter.add(amount, attributes=labels)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
