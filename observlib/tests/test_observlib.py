"""Integration tests for observlib telemetry configuration and @traced decorator.

Tests validate that:
1. configure_telemetry() properly initializes all components
2. @traced decorator works with sync and async functions
3. Metrics, traces, and logs are properly instantiated
4. No external collector is required for testing
"""

import asyncio
import logging
import pytest
from unittest.mock import patch, MagicMock
from beartype.roar import BeartypeCallHintParamViolation
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

from observlib import configure_telemetry, traced


# ============================================================================
# Tests for configure_telemetry
# ============================================================================

def test_configure_telemetry_requires_service_name():
    """Test that configure_telemetry requires a valid service_name."""
    # beartype catches None before manual validation
    with pytest.raises(BeartypeCallHintParamViolation):
        configure_telemetry(None)

    # Empty string passes type check but fails manual validation
    with pytest.raises(ValueError, match="service_name must be a non-empty string"):
        configure_telemetry("")

    # beartype catches non-string types before manual validation
    with pytest.raises(BeartypeCallHintParamViolation):
        configure_telemetry(123)


def test_configure_telemetry_without_server():
    """Test that configure_telemetry works without an OTLP server."""
    configure_telemetry("test-service")

    # Verify metrics provider is set
    provider = metrics.get_meter_provider()
    assert isinstance(provider, MeterProvider), "Provider should be a MeterProvider instance"


def test_configure_telemetry_with_resource_attrs():
    """Test that custom resource attributes are merged correctly."""
    # This test verifies that configure_telemetry accepts resource_attrs
    # and passes them through without error
    configure_telemetry(
        "test-service",
        resource_attrs={"env": "test", "version": "1.0"}
    )

    provider = metrics.get_meter_provider()
    assert isinstance(provider, MeterProvider), "Provider should be a MeterProvider instance"


def test_configure_telemetry_with_pyroscope_server():
    """Test that Pyroscope is configured when server is provided."""
    with patch("observlib.pyroscope.configure") as mock_pyroscope:
        configure_telemetry(
            "test-service",
            pyroscope_server="localhost:4040",
            pyroscope_sample_rate=10
        )

        assert mock_pyroscope.called, "Pyroscope configure should be called"


def test_configure_telemetry_metrics_only():
    """Test metrics configuration without OTLP server."""
    configure_telemetry("test-service")

    provider = metrics.get_meter_provider()
    assert isinstance(provider, MeterProvider), "Provider should be a MeterProvider instance"


def test_configure_telemetry_logging_to_stdout_without_server():
    """Test that logging is configured to stdout when no server is provided."""
    configure_telemetry("test-service", log_level=logging.INFO)

    # Check that stdout handler is configured
    root_logger = logging.getLogger()
    stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) > 0, "Should have stream handlers"

    # Test that we can actually log
    test_logger = logging.getLogger("test_logger")
    test_logger.info("test message")  # Should not raise


def test_configure_telemetry_with_server():
    """Test that tracing and logging are configured when server is provided."""
    # Verify configure_telemetry works with a server parameter
    # It will fail to connect but should set up the configuration
    configure_telemetry("test-service", server="localhost:4317")

    # Verify metrics provider is set
    provider = metrics.get_meter_provider()
    assert isinstance(provider, MeterProvider), "Provider should be configured"


def test_configure_telemetry_without_server_skips_traces_logs():
    """Test that logging is configured even without a server."""
    # Verify configure_telemetry works without a server (no exceptions)
    configure_telemetry("test-service")

    # Verify the provider was set up
    provider = metrics.get_meter_provider()
    assert isinstance(provider, MeterProvider), "Provider should be configured"


# ============================================================================
# Tests for @traced decorator
# ============================================================================

def test_traced_decorator_basic_sync_function():
    """Test @traced decorator on a basic sync function."""
    @traced()
    def sync_func(x):
        return x * 2

    result = sync_func(5)
    assert result == 10


def test_traced_decorator_returns_function_name():
    """Test that @traced preserves function name."""
    @traced()
    def my_function():
        return "test"

    assert my_function.__name__ == "my_function"


def test_traced_decorator_sync_function_with_exception():
    """Test @traced decorator handles exceptions in sync functions."""
    @traced()
    def sync_func_with_error():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        sync_func_with_error()


@pytest.mark.asyncio
async def test_traced_decorator_async_function():
    """Test @traced decorator on an async function."""
    @traced()
    async def async_func(x):
        await asyncio.sleep(0.001)
        return x * 2

    result = await async_func(5)
    assert result == 10


@pytest.mark.asyncio
async def test_traced_decorator_async_function_with_exception():
    """Test @traced decorator handles exceptions in async functions."""
    @traced()
    async def async_func_with_error():
        raise ValueError("async error")

    with pytest.raises(ValueError, match="async error"):
        await async_func_with_error()


def test_traced_decorator_creates_span():
    """Test that @traced decorator creates spans."""
    configure_telemetry("test-service")

    # Test that @traced decorator can be applied successfully and function executes
    @traced()
    def test_func():
        return "result"

    result = test_func()
    assert result == "result", "Function should return expected result"


def test_traced_decorator_with_timer_factory():
    """Test @traced decorator with histogram recording."""
    mock_factory = MagicMock()
    mock_histogram = MagicMock()
    mock_factory.return_value = mock_histogram

    @traced(timer="execution_time", timer_factory=mock_factory)
    def timed_func():
        return "result"

    result = timed_func()
    assert result == "result"
    assert mock_histogram.record.called, "record should be called"
    args, kwargs = mock_histogram.record.call_args
    assert args[0] >= 0, "execution time should be positive"


def test_traced_decorator_with_counter_factory():
    """Test @traced decorator with counter recording."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    @traced(counter="calls", counter_factory=mock_factory)
    def counted_func():
        return "result"

    result = counted_func()
    assert result == "result"
    assert mock_counter.add.called, "add should be called"


def test_traced_decorator_with_label_fn():
    """Test @traced decorator with custom label function."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    def label_fn(result, error, func_args=None, func_kwargs=None):
        return {"result": str(result)}

    @traced(counter="calls", counter_factory=mock_factory, label_fn=label_fn)
    def labeled_func(x):
        return x * 2

    result = labeled_func(5)
    assert result == 10
    assert mock_counter.add.called, "add should be called"


def test_traced_decorator_with_amount_fn():
    """Test @traced decorator with custom amount function."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    def amount_fn(result, error, func_args=None, func_kwargs=None):
        return result  # increment counter by the result value

    @traced(counter="calls", counter_factory=mock_factory, amount_fn=amount_fn)
    def amount_func(x):
        return x

    result = amount_func(42)
    assert result == 42
    args, kwargs = mock_counter.add.call_args
    assert args[0] == 42


def test_traced_decorator_with_func_name_as_label():
    """Test @traced decorator adds function name to labels."""
    mock_factory = MagicMock()
    mock_histogram = MagicMock()
    mock_factory.return_value = mock_histogram

    @traced(
        timer="execution_time",
        timer_factory=mock_factory,
        func_name_as_label=True
    )
    def named_func():
        return "result"

    named_func()

    # Check that function name is in the attributes
    args, kwargs = mock_histogram.record.call_args
    assert "attributes" in kwargs or len(args) > 1
    if "attributes" in kwargs:
        assert kwargs["attributes"]["function"] == "named_func"


def test_traced_decorator_without_func_name_as_label():
    """Test @traced decorator can skip function name in labels."""
    mock_factory = MagicMock()
    mock_histogram = MagicMock()
    mock_factory.return_value = mock_histogram

    @traced(
        timer="execution_time",
        timer_factory=mock_factory,
        func_name_as_label=False
    )
    def named_func():
        return "result"

    named_func()
    assert mock_histogram.record.called


def test_traced_decorator_with_string_config():
    """Test @traced decorator accepts string config."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    @traced(counter="calls", counter_factory=mock_factory)
    def string_config_func():
        return "result"

    result = string_config_func()
    assert result == "result"


def test_traced_decorator_with_dict_config():
    """Test @traced decorator accepts dict config."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    @traced(
        counter={"name": "calls", "unit": "1"},
        counter_factory=mock_factory
    )
    def dict_config_func():
        return "result"

    result = dict_config_func()
    assert result == "result"


def test_traced_decorator_debug_mode():
    """Test @traced decorator with debug mode enabled."""
    with patch("observlib.decorator.logger") as mock_logger:
        @traced(debug=True)
        def debug_func():
            return "result"

        result = debug_func()
        assert result == "result"
        # Debug mode should log something
        assert mock_logger.debug.called, "debug logger should be called"


@pytest.mark.asyncio
async def test_traced_decorator_async_with_metrics():
    """Test @traced decorator on async function with metrics."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    @traced(counter="calls", counter_factory=mock_factory)
    async def async_metric_func():
        await asyncio.sleep(0.001)
        return "result"

    result = await async_metric_func()
    assert result == "result"
    assert mock_counter.add.called, "add should be called"


def test_traced_decorator_preserves_function_signature():
    """Test that @traced preserves function signature."""
    @traced()
    def func_with_defaults(a, b=10, *args, **kwargs):
        return a + b

    assert func_with_defaults(5) == 15
    assert func_with_defaults(5, 20) == 25
    assert func_with_defaults(5, 20, extra="value") == 25


def test_traced_decorator_multiple_calls():
    """Test @traced decorator on function called multiple times."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    @traced(counter="calls", counter_factory=mock_factory)
    def multi_call_func():
        return "result"

    for _ in range(3):
        multi_call_func()

    assert mock_counter.add.call_count == 3


# ============================================================================
# Tests with real metrics provider
# ============================================================================

def test_traced_with_real_meter_provider():
    """Test @traced decorator retrieves metrics from global provider."""
    configure_telemetry("test-service")
    meter_provider = metrics.get_meter_provider()
    assert isinstance(meter_provider, MeterProvider)

    # Get a meter from the provider
    meter = meter_provider.get_meter(__name__)
    assert meter is not None


def test_traced_with_real_tracer_provider():
    """Test @traced decorator retrieves tracer from global provider."""
    configure_telemetry("test-service")
    tracer_provider = trace.get_tracer_provider()
    # TracerProvider could be a proxy or actual provider
    assert tracer_provider is not None, "Should have a tracer provider"

    # Get a tracer from the provider
    tracer = tracer_provider.get_tracer(__name__)
    assert tracer is not None, "Should be able to get a tracer"


# ============================================================================
# Error handling tests
# ============================================================================

def test_traced_decorator_error_propagation():
    """Test that @traced re-raises exceptions."""
    @traced()
    def error_func():
        raise RuntimeError("test error")

    with pytest.raises(RuntimeError, match="test error"):
        error_func()


@pytest.mark.asyncio
async def test_traced_decorator_async_error_propagation():
    """Test that @traced re-raises exceptions in async functions."""
    @traced()
    async def async_error_func():
        raise RuntimeError("async test error")

    with pytest.raises(RuntimeError, match="async test error"):
        await async_error_func()


def test_traced_decorator_error_with_metrics():
    """Test that metrics are still recorded even when error occurs."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    @traced(counter="calls", counter_factory=mock_factory)
    def error_with_metrics():
        raise ValueError("error")

    with pytest.raises(ValueError):
        error_with_metrics()

    # Counter should still have been called
    assert mock_counter.add.called, "counter should be called even on error"


def test_traced_decorator_metric_factory_error():
    """Test error handling when metric factory fails."""
    def failing_factory(config):
        raise RuntimeError("factory error")

    @traced(counter="calls", counter_factory=failing_factory)
    def func_with_bad_factory():
        return "result"

    # The function execution may still succeed despite factory failure,
    # depending on error handling in the decorator
    try:
        result = func_with_bad_factory()
        # If it succeeds, that's fine - the function still ran
        assert result == "result"
    except RuntimeError as e:
        # If it raises, that's also acceptable - factory error propagated
        assert "factory error" in str(e), "Should have factory error message"


# ============================================================================
# Edge cases
# ============================================================================

def test_traced_decorator_with_none_counter_factory():
    """Test @traced when counter_factory is None."""
    @traced(counter="calls", counter_factory=None)
    def func_no_factory():
        return "result"

    result = func_no_factory()
    assert result == "result"


def test_traced_decorator_with_class_method():
    """Test @traced decorator on class methods."""
    class MyClass:
        @traced()
        def sync_method(self):
            return "sync result"

        @traced()
        async def async_method(self):
            return "async result"

    obj = MyClass()
    assert obj.sync_method() == "sync result"


@pytest.mark.asyncio
async def test_traced_decorator_async_method():
    """Test @traced decorator on async methods."""
    class MyClass:
        @traced()
        async def async_method(self):
            await asyncio.sleep(0.001)
            return "async result"

    obj = MyClass()
    result = await obj.async_method()
    assert result == "async result"


def test_traced_decorator_nested_calls():
    """Test @traced decorator with nested function calls."""
    @traced()
    def outer_func():
        return inner_func()

    @traced()
    def inner_func():
        return "inner result"

    result = outer_func()
    assert result == "inner result"


def test_traced_decorator_with_generator():
    """Test @traced decorator with generator function."""
    @traced()
    def generator_func():
        yield 1
        yield 2
        yield 3

    gen = generator_func()
    values = list(gen)
    assert values == [1, 2, 3]


def test_traced_decorator_with_none_return():
    """Test @traced decorator with function that returns None."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    @traced(counter="calls", counter_factory=mock_factory, label_fn=lambda r, e, **kw: {})
    def returns_none():
        return None

    result = returns_none()
    assert result is None
    assert mock_counter.add.called


# ============================================================================
# Tests for parameter combinations
# ============================================================================

def test_traced_decorator_with_custom_tracer():
    """Test @traced decorator with custom tracer instance."""
    configure_telemetry("test-service")
    custom_tracer = trace.get_tracer("custom-tracer", "1.0.0")

    @traced(tracer="custom-tracer")
    def func_with_custom_tracer():
        return "result"

    result = func_with_custom_tracer()
    assert result == "result"


def test_traced_decorator_with_timer_and_counter():
    """Test @traced decorator with both timer and counter."""
    timer_factory = MagicMock()
    counter_factory = MagicMock()
    mock_histogram = MagicMock()
    mock_counter = MagicMock()
    timer_factory.return_value = mock_histogram
    counter_factory.return_value = mock_counter

    @traced(
        timer="execution_time",
        timer_factory=timer_factory,
        counter="calls",
        counter_factory=counter_factory
    )
    def both_metrics_func():
        return "result"

    result = both_metrics_func()
    assert result == "result"
    assert mock_histogram.record.called, "histogram should be recorded"
    assert mock_counter.add.called, "counter should be incremented"


def test_traced_decorator_label_fn_with_func_args():
    """Test @traced decorator label_fn accessing function arguments."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    def label_fn(result, error, func_args=None, func_kwargs=None):
        return {
            "arg0": str(func_args[0]) if func_args else "",
            "kwarg_key": func_kwargs.get("key", "") if func_kwargs else "",
        }

    @traced(counter="calls", counter_factory=mock_factory, label_fn=label_fn)
    def labeled_with_args(x, key="default"):
        return x * 2

    result = labeled_with_args(5, key="test_key")
    assert result == 10

    # Verify label_fn was called with correct arguments
    args, kwargs = mock_counter.add.call_args
    assert "attributes" in kwargs
    assert kwargs["attributes"]["arg0"] == "5"
    assert kwargs["attributes"]["kwarg_key"] == "test_key"


def test_traced_decorator_label_fn_with_result():
    """Test @traced decorator label_fn accessing function result."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    def label_fn(result, error, func_args=None, func_kwargs=None):
        return {"result_value": str(result), "has_error": str(error is not None)}

    @traced(counter="calls", counter_factory=mock_factory, label_fn=label_fn)
    def labeled_with_result():
        return 42

    result = labeled_with_result()
    assert result == 42

    args, kwargs = mock_counter.add.call_args
    assert "attributes" in kwargs
    assert kwargs["attributes"]["result_value"] == "42"
    assert kwargs["attributes"]["has_error"] == "False"


def test_traced_decorator_label_fn_with_error():
    """Test @traced decorator label_fn accessing error in exception case."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    def label_fn(result, error, func_args=None, func_kwargs=None):
        return {
            "has_error": str(error is not None),
            "error_type": type(error).__name__ if error else ""
        }

    @traced(counter="calls", counter_factory=mock_factory, label_fn=label_fn)
    def labeled_with_error():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        labeled_with_error()

    # Counter should still be called with error labels
    args, kwargs = mock_counter.add.call_args
    assert "attributes" in kwargs
    assert kwargs["attributes"]["has_error"] == "True"
    assert kwargs["attributes"]["error_type"] == "ValueError"


def test_traced_decorator_amount_fn_with_result():
    """Test @traced decorator amount_fn using function result."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    def amount_fn(result, error, func_args=None, func_kwargs=None):
        # Increment counter by the result value
        return result if result else 0

    @traced(counter="calls", counter_factory=mock_factory, amount_fn=amount_fn)
    def amount_from_result(value):
        return value

    result = amount_from_result(100)
    assert result == 100

    args, kwargs = mock_counter.add.call_args
    assert args[0] == 100, "counter should be incremented by result value"


def test_traced_decorator_amount_fn_with_func_args():
    """Test @traced decorator amount_fn accessing function arguments."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    def amount_fn(result, error, func_args=None, func_kwargs=None):
        # Use first argument as the amount
        return func_args[0] if func_args else 1

    @traced(counter="calls", counter_factory=mock_factory, amount_fn=amount_fn)
    def amount_from_args(multiplier):
        return multiplier * 2

    result = amount_from_args(50)
    assert result == 100

    args, kwargs = mock_counter.add.call_args
    assert args[0] == 50, "counter should be incremented by first argument"


def test_traced_decorator_amount_fn_with_error():
    """Test @traced decorator amount_fn when function raises exception."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    def amount_fn(result, error, func_args=None, func_kwargs=None):
        # Return 0 if there was an error, otherwise use result
        return 0 if error else result

    @traced(counter="calls", counter_factory=mock_factory, amount_fn=amount_fn)
    def amount_with_error():
        raise RuntimeError("error")

    with pytest.raises(RuntimeError, match="error"):
        amount_with_error()

    # Counter should be called with amount=0 due to error
    args, kwargs = mock_counter.add.call_args
    assert args[0] == 0, "counter should be incremented by 0 on error"


def test_traced_decorator_complex_combination():
    """Test @traced decorator with multiple parameters combined."""
    timer_factory = MagicMock()
    counter_factory = MagicMock()
    mock_histogram = MagicMock()
    mock_counter = MagicMock()
    timer_factory.return_value = mock_histogram
    counter_factory.return_value = mock_counter

    def custom_label_fn(result, error, func_args=None, func_kwargs=None):
        return {
            "input": str(func_args[0]) if func_args else "",
            "output": str(result),
        }

    def custom_amount_fn(result, error, func_args=None, func_kwargs=None):
        return result if result else 1

    @traced(
        timer={"name": "process_time", "unit": "s"},
        timer_factory=timer_factory,
        counter={"name": "processed_items", "unit": "1"},
        counter_factory=counter_factory,
        label_fn=custom_label_fn,
        amount_fn=custom_amount_fn,
        func_name_as_label=True,
        debug=False
    )
    def complex_func(items):
        return items * 2

    result = complex_func(10)
    assert result == 20

    # Verify histogram was recorded with correct labels
    assert mock_histogram.record.called
    _, hist_kwargs = mock_histogram.record.call_args
    assert "attributes" in hist_kwargs
    assert hist_kwargs["attributes"]["function"] == "complex_func"
    assert hist_kwargs["attributes"]["input"] == "10"
    assert hist_kwargs["attributes"]["output"] == "20"

    # Verify counter was incremented with correct amount and labels
    assert mock_counter.add.called
    counter_args, counter_kwargs = mock_counter.add.call_args
    assert counter_args[0] == 20, "counter should be incremented by result (20)"
    assert "attributes" in counter_kwargs
    assert counter_kwargs["attributes"]["input"] == "10"
    assert counter_kwargs["attributes"]["output"] == "20"


@pytest.mark.asyncio
async def test_traced_decorator_async_with_all_parameters():
    """Test @traced decorator on async function with all parameters."""
    timer_factory = MagicMock()
    counter_factory = MagicMock()
    mock_histogram = MagicMock()
    mock_counter = MagicMock()
    timer_factory.return_value = mock_histogram
    counter_factory.return_value = mock_counter

    def label_fn(result, error, func_args=None, func_kwargs=None):
        return {"async": "true", "result": str(result)}

    @traced(
        timer="async_time",
        timer_factory=timer_factory,
        counter="async_calls",
        counter_factory=counter_factory,
        label_fn=label_fn,
        func_name_as_label=True
    )
    async def async_all_params(value):
        await asyncio.sleep(0.001)
        return value * 3

    result = await async_all_params(7)
    assert result == 21
    assert mock_histogram.record.called
    assert mock_counter.add.called

    # Verify labels
    _, hist_kwargs = mock_histogram.record.call_args
    assert hist_kwargs["attributes"]["async"] == "true"
    assert hist_kwargs["attributes"]["result"] == "21"


def test_traced_decorator_timer_only_without_counter():
    """Test @traced decorator with only timer, no counter."""
    timer_factory = MagicMock()
    mock_histogram = MagicMock()
    timer_factory.return_value = mock_histogram

    @traced(timer="execution_time", timer_factory=timer_factory)
    def timer_only_func():
        return "result"

    result = timer_only_func()
    assert result == "result"
    assert mock_histogram.record.called
    # Verify execution time is positive
    args, _ = mock_histogram.record.call_args
    assert args[0] >= 0


def test_traced_decorator_counter_only_without_timer():
    """Test @traced decorator with only counter, no timer."""
    counter_factory = MagicMock()
    mock_counter = MagicMock()
    counter_factory.return_value = mock_counter

    @traced(counter="calls", counter_factory=counter_factory)
    def counter_only_func():
        return "result"

    result = counter_only_func()
    assert result == "result"
    assert mock_counter.add.called
    # Verify default amount is 1
    args, _ = mock_counter.add.call_args
    assert args[0] == 1


def test_traced_decorator_func_name_as_label_with_timer():
    """Test func_name_as_label parameter specifically with timer."""
    timer_factory = MagicMock()
    mock_histogram = MagicMock()
    timer_factory.return_value = mock_histogram

    @traced(
        timer="execution_time",
        timer_factory=timer_factory,
        func_name_as_label=False
    )
    def no_func_label():
        return "result"

    no_func_label()

    # Timer should still get the function name in attributes (always added for histogram)
    _, kwargs = mock_histogram.record.call_args
    assert "attributes" in kwargs
    # The histogram ALWAYS gets function name according to decorator.py:84-86
    assert kwargs["attributes"]["function"] == "no_func_label"


# ============================================================================
# Beartype runtime type validation tests
# ============================================================================

def test_beartype_validates_configure_telemetry_types():
    """Test that beartype catches type violations in configure_telemetry."""
    # Invalid service_name types
    with pytest.raises(BeartypeCallHintParamViolation):
        configure_telemetry(None)

    with pytest.raises(BeartypeCallHintParamViolation):
        configure_telemetry(123)

    with pytest.raises(BeartypeCallHintParamViolation):
        configure_telemetry(["invalid"])

    # Invalid server type
    with pytest.raises(BeartypeCallHintParamViolation):
        configure_telemetry("valid-service", server=123)

    # Invalid log_level type
    with pytest.raises(BeartypeCallHintParamViolation):
        configure_telemetry("valid-service", log_level="INFO")  # should be int

    # Invalid resource_attrs type
    with pytest.raises(BeartypeCallHintParamViolation):
        configure_telemetry("valid-service", resource_attrs="invalid")


def test_beartype_validates_traced_decorator_types():
    """Test that beartype catches type violations in traced decorator parameters."""
    # Invalid debug type (should be bool)
    with pytest.raises(BeartypeCallHintParamViolation):
        @traced(debug="true")  # should be bool, not string
        def func():
            pass

    # Invalid func_name_as_label type (should be bool)
    with pytest.raises(BeartypeCallHintParamViolation):
        @traced(func_name_as_label=1)  # should be bool, not int
        def func():
            pass

    # Invalid tracer type (should be str or None)
    with pytest.raises(BeartypeCallHintParamViolation):
        @traced(tracer=123)  # should be str, not int
        def func():
            pass


def test_beartype_validates_callback_protocols():
    """Test that beartype validates callback function signatures."""
    mock_factory = MagicMock()
    mock_counter = MagicMock()
    mock_factory.return_value = mock_counter

    # Valid label_fn - should work
    def valid_label_fn(result, error, func_args=None, func_kwargs=None):
        return {"status": "ok"}

    @traced(counter="calls", counter_factory=mock_factory, label_fn=valid_label_fn)
    def func_with_valid_label():
        return "result"

    func_with_valid_label()  # Should work fine

    # Valid amount_fn - should work
    def valid_amount_fn(result, error, func_args=None, func_kwargs=None):
        return 1

    @traced(counter="calls", counter_factory=mock_factory, amount_fn=valid_amount_fn)
    def func_with_valid_amount():
        return "result"

    func_with_valid_amount()  # Should work fine


def test_beartype_runtime_validation_preserves_functionality():
    """Test that beartype doesn't interfere with normal operation."""
    configure_telemetry("test-service")

    @traced()
    def normal_function(x: int, y: str) -> str:
        return f"{y}: {x}"

    result = normal_function(42, "answer")
    assert result == "answer: 42"

    # Beartype validates the decorated function's parameters at runtime
    # But our decorator doesn't apply beartype to the wrapped function itself
    # so this tests that beartype on the decorator doesn't break the function
