from functools import wraps
from typing import TypeVar, ParamSpec, Optional, Any, Protocol, runtime_checkable, TypeAlias
from beartype import beartype
from beartype.typing import Callable, Union, Awaitable
from opentelemetry.sdk.trace import Status, StatusCode
from opentelemetry import trace
import time
import asyncio
import logging

logger: logging.Logger = logging.getLogger(__name__)

# Type variables for generic function signatures
P = ParamSpec('P')
R = TypeVar('R')

# Type aliases for clarity
MetricConfig: TypeAlias = Union[str, dict[str, Any]]
Labels: TypeAlias = dict[str, str]


@runtime_checkable
class MetricFactory(Protocol):
    """Protocol for metric factory callables."""
    def __call__(self, config: frozenset[tuple[str, Any]]) -> Any:
        """Create a metric instance from frozen config."""
        ...


@runtime_checkable
class LabelFunction(Protocol):
    """Protocol for label generation functions."""
    def __call__(
        self,
        result: Any,
        error: Optional[Exception],
        func_args: Optional[tuple[Any, ...]] = None,
        func_kwargs: Optional[dict[str, Any]] = None,
    ) -> Labels:
        """Generate labels from function execution context."""
        ...


@runtime_checkable
class AmountFunction(Protocol):
    """Protocol for amount calculation functions."""
    def __call__(
        self,
        result: Any,
        error: Optional[Exception],
        func_args: Optional[tuple[Any, ...]] = None,
        func_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union[int, float]:
        """Calculate counter increment amount from execution context."""
        ...


@beartype
def _normalize_metric_config(config: MetricConfig) -> dict[str, Any]:
    """Convert a metric config to a normalized dictionary.

    Args:
        config: Either a string (metric name) or a dict of config options

    Returns:
        A dictionary with the normalized config
    """
    if isinstance(config, (str, bytes)):
        return {"name": config}
    return config  # type: ignore[no-any-return]


@beartype
def traced(
    timer: Optional[MetricConfig] = None,
    timer_factory: Optional[MetricFactory] = None,
    counter: Optional[MetricConfig] = None,
    counter_factory: Optional[MetricFactory] = None,
    label_fn: Optional[LabelFunction] = None,
    amount_fn: Optional[AmountFunction] = None,
    tracer: Optional[str] = None,
    debug: bool = False,
    func_name_as_label: bool = True,
) -> Callable[[Callable[P, R] | Callable[P, Awaitable[R]]], Callable[P, R] | Callable[P, Awaitable[R]]]:
    def decorator(func: Callable[P, R] | Callable[P, Awaitable[R]]) -> Callable[P, R] | Callable[P, Awaitable[R]]:
        def record_data(
            func_name: str,
            timer: Optional[MetricConfig],
            counter: Optional[MetricConfig],
            counter_factory: Optional[MetricFactory],
            label_fn: Optional[LabelFunction],
            amount_fn: Optional[AmountFunction],
            start_time: float,
            result: Any,
            error: Optional[Exception],
            debug: bool,
            func_name_as_label: bool,
            func_args: tuple[Any, ...],
            func_kwargs: dict[str, Any],
        ) -> None:
            labels = (
                label_fn(
                    result,
                    error,
                    func_args=func_args,
                    func_kwargs=func_kwargs,
                )
                if label_fn
                else {}
            )

            if func_name_as_label:
                labels["function"] = func_name
            if debug:
                logger.debug(f"labels: {labels}")

            if timer and callable(timer_factory):
                config = _normalize_metric_config(timer)

                if debug:
                    logger.debug(f"requesting histogram with {config}")

                try:
                    exec_time_histogram = timer_factory(frozenset(config.items()))
                except Exception as ex:
                    if debug:
                        logger.debug(f"error requesting histogram: {ex}")
                    raise

                exec_time_histogram.record(
                    time.perf_counter() - start_time,
                    attributes=labels
                    | {
                        "function": func_name
                    },  # histogram always gets the function name
                )

            amount = (
                amount_fn(
                    result,
                    error,
                    func_args=func_args,
                    func_kwargs=func_kwargs,
                )
                if amount_fn
                else 1
            )
            if debug:
                logger.debug(f"amount: {amount}")

            if counter and callable(counter_factory):
                config = _normalize_metric_config(counter)

                if debug:
                    logger.debug(f"requesting counter with config: {config}")

                try:
                    actual_counter = counter_factory(frozenset(config.items()))
                except Exception as ex:
                    if debug:
                        logger.debug(f"error requesting counter: {ex}")
                    raise

                actual_counter.add(amount, attributes=labels)

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if debug:
                logger.debug(f"called sync wrapper with:\nargs:{args}\nkwargs:{kwargs}")
            start = time.perf_counter()
            with trace.get_tracer(tracer).start_as_current_span(func.__name__) as span:
                result: Any = None
                error: Optional[Exception] = None
                try:
                    result = func(*args, **kwargs)
                    return result  # type: ignore[no-any-return]
                except Exception as ex:
                    error = ex
                    span.set_status(Status(StatusCode.ERROR))
                    raise

                finally:
                    try:
                        record_data(
                            func.__name__,
                            timer,
                            counter,
                            counter_factory,
                            label_fn,
                            amount_fn,
                            start,
                            result,
                            error,
                            debug,
                            func_name_as_label,
                            args,
                            kwargs,
                        )
                    except Exception as ex:
                        if debug:
                            logger.debug(f"exception recording data: {ex}")

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if debug:
                logger.debug(f"called async wrapper with:\nargs:{args}\nkwargs:{kwargs}")
            start = time.perf_counter()
            with trace.get_tracer(tracer).start_as_current_span(func.__name__) as span:
                result: Any = None
                error: Optional[Exception] = None
                try:
                    result = await func(*args, **kwargs)
                    return result  # type: ignore[no-any-return]

                except Exception as ex:
                    error = ex
                    span.set_status(Status(StatusCode.ERROR))
                    raise

                finally:
                    try:
                        record_data(
                            func.__name__,
                            timer,
                            counter,
                            counter_factory,
                            label_fn,
                            amount_fn,
                            start,
                            result,
                            error,
                            debug,
                            func_name_as_label,
                            args,
                            kwargs,
                        )
                    except Exception as ex:
                        if debug:
                            logger.debug(f"exception recording data: {ex}")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
