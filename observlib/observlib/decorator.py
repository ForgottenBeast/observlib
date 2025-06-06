from functools import wraps
from opentelemetry.sdk.trace import Status, StatusCode
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
    debug=False,
    func_name_as_label=False,
):
    def decorator(func):
        def record_data(
            func_name,
            timing_histogram,
            counter,
            counter_factory,
            label_fn,
            amount_fn,
            start_time,
            result,
            error,
            debug,
            func_name_as_label,
        ):
            if timing_histogram and callable(timer_factory):
                if isinstance(timing_histogram, (str, bytes)):
                    config = {"name": timing_histogram}
                else:
                    config = timing_histogram

                if debug:
                    print(f"requesting histogram with {config}")

                try:
                    exec_time_histogram = timer_factory(frozenset(config.items()))
                except Exception as ex:
                    if debug:
                        print(f"error requesting histogram: {ex}")
                    raise

                exec_time_histogram.record(
                    time.perf_counter() - start_time, attributes={"function": func_name}
                )

            labels = label_fn(result, error) if label_fn else {}

            if func_name_as_label:
                labels["func"] = func_name

            if debug:
                print(f"labels: {labels}")

            amount = amount_fn(result, error) if amount_fn else 1
            if debug:
                print(f"amount: {amount}")

            if counter and callable(counter_factory):
                if isinstance(counter, (str, bytes)):
                    config = {"name": counter}
                else:
                    config = counter

                if debug:
                    print(f"requesting counter with config: {config}")

                try:
                    actual_counter = counter_factory(frozenset(config.items()))
                except Exception as ex:
                    if debug:
                        print(f"error requesting counter: {ex}")
                    raise

                actual_counter.add(amount, attributes=labels)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if debug:
                print(f"called sync wrapper with:\nargs:{args}\nkwargs:{kwargs}")
            start = time.perf_counter()
            with trace.get_tracer(tracer).start_as_current_span(func.__name__) as span:
                result = None
                error = None
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as ex:
                    error = ex
                    span.set_status(Status(StatusCode.ERROR))
                    raise

                finally:
                    try:
                        record_data(
                            func.__name__,
                            timing_histogram,
                            counter,
                            counter_factory,
                            label_fn,
                            amount_fn,
                            start,
                            result,
                            error,
                            debug,
                            func_name_as_label,
                        )
                    except Exception as ex:
                        if debug:
                            print(f"exception recording data: {ex}")

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if debug:
                print(f"called async wrapper with:\nargs:{args}\nkwargs:{kwargs}")
            start = time.perf_counter()
            with trace.get_tracer(tracer).start_as_current_span(func.__name__) as span:
                result = None
                error = None
                try:
                    result = await func(*args, **kwargs)
                    return result

                except Exception as ex:
                    error = ex
                    span.set_status(Status(StatusCode.ERROR))
                    raise

                finally:
                    try:
                        record_data(
                            func.__name__,
                            timing_histogram,
                            counter,
                            counter_factory,
                            label_fn,
                            amount_fn,
                            start,
                            result,
                            error,
                            debug,
                            func_name_as_label,
                        )
                    except Exception as ex:
                        if debug:
                            print(f"exception recording data: {ex}")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
