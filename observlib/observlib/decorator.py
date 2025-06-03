from functools import wraps
from opentelemetry import trace
import time
import asyncio


def traced(
    timing_histogram=None,
    counter=None,
    counter_factory=None,
    timer_factory=None,
    label_fn=None,
    amount_fn=None,
    tracer=None,
):

    def decorator(func):
        def record_data(func_name, timing_histogram, counter, counter_factory,label_fn, amount_fn, start_time,result, error):
            if timing_histogram and callable(timer_factory):
                if isinstance(timing_histogram, (str, bytes)):
                    config = {"name": timing_histogram}
                else:
                    config = timing_histogram

                exec_time_histogram = timer_factory(frozenset(config.items()))
                exec_time_histogram.record(
                    time.perf_counter() - start_time, attributes = {"function": func_name}
                )

            labels = (
                label_fn(result=result, exception=error) if label_fn else {}
            )
            amount = (
                amount_fn(result=result, exception=error) if amount_fn else 1
            )

            if counter and callable(counter_factory):
                if isinstance(counter, (str, bytes)):
                    config = {"name": counter}
                else:
                    config = counter

                actual_counter = counter_factory(frozenset(config.items()))
                actual_counter.add(amount, attributes=labels)

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
                    record_data(func.__name__, timing_histogram, counter, counter_factory,label_fn, amount_fn, start, result, error)

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
                    record_data(func.__name__, timing_histogram, counter, counter_factory,label_fn, amount_fn, start, result, error)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
