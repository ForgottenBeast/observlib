"""Integration tests for observlib telemetry configuration and @traced decorator.

These tests validate that:
1. configure_telemetry() properly initializes all components
2. @traced decorator works with sync and async functions
3. Metrics, traces, and logs are properly instantiated
4. No external collector is required for testing
"""

import asyncio
import logging
import pytest
from unittest.mock import patch, MagicMock, call
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

from observlib import configure_telemetry, traced


class TestConfigureTelemetry:
    """Test suite for configure_telemetry function."""

    def test_configure_telemetry_requires_service_name(self):
        """Test that configure_telemetry requires a valid service_name."""
        with pytest.raises(ValueError, match="service_name must be a non-empty string"):
            configure_telemetry(None)

        with pytest.raises(ValueError, match="service_name must be a non-empty string"):
            configure_telemetry("")

        with pytest.raises(ValueError, match="service_name must be a non-empty string"):
            configure_telemetry(123)

    def test_configure_telemetry_without_server(self):
        """Test that configure_telemetry works without an OTLP server."""
        configure_telemetry("test-service")

        # Verify metrics provider is set
        provider = metrics.get_meter_provider()
        assert isinstance(provider, MeterProvider)
        assert provider.resource.attributes["service.name"] == "test-service"

    def test_configure_telemetry_with_resource_attrs(self):
        """Test that custom resource attributes are merged correctly."""
        configure_telemetry(
            "test-service",
            resource_attrs={"env": "test", "version": "1.0"}
        )

        provider = metrics.get_meter_provider()
        attrs = provider.resource.attributes
        assert attrs["service.name"] == "test-service"
        assert attrs["env"] == "test"
        assert attrs["version"] == "1.0"

    def test_configure_telemetry_with_pyroscope_server(self):
        """Test that Pyroscope is configured when server is provided."""
        with patch("observlib.pyroscope.configure") as mock_pyroscope:
            configure_telemetry(
                "test-service",
                pyroscope_server="localhost:4040",
                pyroscope_sample_rate=10
            )

            mock_pyroscope.assert_called_once_with(
                application_name="test-service",
                server_address="http://localhost:4040",
                sample_rate=10,
            )

    def test_configure_telemetry_metrics_only(self):
        """Test metrics configuration without OTLP server."""
        configure_telemetry("test-service")

        provider = metrics.get_meter_provider()
        assert isinstance(provider, MeterProvider)
        # Prometheus reader should be available
        assert len(provider._metric_readers) >= 1

    def test_configure_telemetry_logging_to_stdout_without_server(self):
        """Test that logging is configured to stdout when no server is provided."""
        configure_telemetry("test-service", log_level=logging.INFO)

        # Check that stdout handler is configured
        root_logger = logging.getLogger()
        stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) > 0

        # Test that we can actually log
        test_logger = logging.getLogger("test_logger")
        test_logger.info("test message")  # Should not raise

    @patch("observlib.traces.configure_tracing")
    @patch("observlib.logs.configure_logging")
    def test_configure_telemetry_with_server(self, mock_logs, mock_traces):
        """Test that tracing and logging are configured when server is provided."""
        configure_telemetry("test-service", server="localhost:4317")

        mock_traces.assert_called_once()
        mock_logs.assert_called_once()

        # Verify the resource was passed
        call_args = mock_traces.call_args
        assert call_args[0][0] == "localhost:4317"
        assert call_args[0][1].attributes["service.name"] == "test-service"

    @patch("observlib.traces.configure_tracing")
    @patch("observlib.logs.configure_logging")
    def test_configure_telemetry_without_server_skips_traces_logs(self, mock_logs, mock_traces):
        """Test that tracing is skipped but logging is always configured."""
        configure_telemetry("test-service")

        mock_traces.assert_not_called()
        # Logging should still be called even without a server (will use stdout)
        mock_logs.assert_called_once()


class TestTracedDecorator:
    """Test suite for @traced decorator."""

    def test_traced_decorator_basic_sync_function(self):
        """Test @traced decorator on a basic sync function."""
        @traced()
        def sync_func(x):
            return x * 2

        result = sync_func(5)
        assert result == 10

    def test_traced_decorator_returns_function_name(self):
        """Test that @traced preserves function name."""
        @traced()
        def my_function():
            return "test"

        assert my_function.__name__ == "my_function"

    def test_traced_decorator_sync_function_with_exception(self):
        """Test @traced decorator handles exceptions in sync functions."""
        @traced()
        def sync_func_with_error():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            sync_func_with_error()

    @pytest.mark.asyncio
    async def test_traced_decorator_async_function(self):
        """Test @traced decorator on an async function."""
        @traced()
        async def async_func(x):
            await asyncio.sleep(0.001)
            return x * 2

        result = await async_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_traced_decorator_async_function_with_exception(self):
        """Test @traced decorator handles exceptions in async functions."""
        @traced()
        async def async_func_with_error():
            raise ValueError("async error")

        with pytest.raises(ValueError, match="async error"):
            await async_func_with_error()

    def test_traced_decorator_creates_span(self):
        """Test that @traced decorator creates spans."""
        configure_telemetry("test-service")
        tracer = trace.get_tracer(__name__)

        spans = []

        class SpanCapture:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def set_status(self, status):
                pass

        original_start = tracer.start_as_current_span

        def capture_span(name):
            spans.append(name)
            return original_start(name)

        with patch.object(tracer, 'start_as_current_span', side_effect=capture_span):
            @traced(tracer=tracer)
            def test_func():
                return "result"

            test_func()

        assert "test_func" in spans

    def test_traced_decorator_with_timer_factory(self):
        """Test @traced decorator with histogram recording."""
        mock_factory = MagicMock()
        mock_histogram = MagicMock()
        mock_factory.return_value = mock_histogram

        @traced(timer="execution_time", timer_factory=mock_factory)
        def timed_func():
            return "result"

        result = timed_func()
        assert result == "result"
        mock_histogram.record.assert_called_once()
        args, kwargs = mock_histogram.record.call_args
        assert args[0] >= 0  # execution time should be positive

    def test_traced_decorator_with_counter_factory(self):
        """Test @traced decorator with counter recording."""
        mock_factory = MagicMock()
        mock_counter = MagicMock()
        mock_factory.return_value = mock_counter

        @traced(counter="calls", counter_factory=mock_factory)
        def counted_func():
            return "result"

        result = counted_func()
        assert result == "result"
        mock_counter.add.assert_called_once()

    def test_traced_decorator_with_label_fn(self):
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
        mock_counter.add.assert_called_once()

    def test_traced_decorator_with_amount_fn(self):
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

    def test_traced_decorator_with_func_name_as_label(self):
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

    def test_traced_decorator_without_func_name_as_label(self):
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
        mock_histogram.record.assert_called_once()

    def test_traced_decorator_with_string_config(self):
        """Test @traced decorator accepts string config."""
        mock_factory = MagicMock()
        mock_counter = MagicMock()
        mock_factory.return_value = mock_counter

        @traced(counter="calls", counter_factory=mock_factory)
        def string_config_func():
            return "result"

        result = string_config_func()
        assert result == "result"

    def test_traced_decorator_with_dict_config(self):
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

    def test_traced_decorator_debug_mode(self):
        """Test @traced decorator with debug mode enabled."""
        with patch("observlib.decorator.logger") as mock_logger:
            @traced(debug=True)
            def debug_func():
                return "result"

            result = debug_func()
            assert result == "result"
            # Debug mode should log something
            assert mock_logger.debug.called

    @pytest.mark.asyncio
    async def test_traced_decorator_async_with_metrics(self):
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
        mock_counter.add.assert_called_once()

    def test_traced_decorator_preserves_function_signature(self):
        """Test that @traced preserves function signature."""
        @traced()
        def func_with_defaults(a, b=10, *args, **kwargs):
            return a + b

        assert func_with_defaults(5) == 15
        assert func_with_defaults(5, 20) == 25
        assert func_with_defaults(5, 20, extra="value") == 25

    def test_traced_decorator_multiple_calls(self):
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


class TestTracedDecoratorWithGlobalMetrics:
    """Test @traced decorator with actual global metrics provider."""

    def setup_method(self):
        """Set up test with metrics provider."""
        configure_telemetry("test-service")

    def test_traced_with_real_meter_provider(self):
        """Test @traced decorator retrieves metrics from global provider."""
        meter_provider = metrics.get_meter_provider()
        assert isinstance(meter_provider, MeterProvider)

        # Get a meter from the provider
        meter = meter_provider.get_meter(__name__)
        assert meter is not None

    def test_traced_with_real_tracer_provider(self):
        """Test @traced decorator retrieves tracer from global provider."""
        tracer_provider = trace.get_tracer_provider()
        assert isinstance(tracer_provider, TracerProvider)

        # Get a tracer from the provider
        tracer = tracer_provider.get_tracer(__name__)
        assert tracer is not None


class TestErrorHandling:
    """Test error handling in @traced decorator."""

    def test_traced_decorator_error_propagation(self):
        """Test that @traced re-raises exceptions."""
        @traced()
        def error_func():
            raise RuntimeError("test error")

        with pytest.raises(RuntimeError, match="test error"):
            error_func()

    @pytest.mark.asyncio
    async def test_traced_decorator_async_error_propagation(self):
        """Test that @traced re-raises exceptions in async functions."""
        @traced()
        async def async_error_func():
            raise RuntimeError("async test error")

        with pytest.raises(RuntimeError, match="async test error"):
            await async_error_func()

    def test_traced_decorator_error_with_metrics(self):
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
        mock_counter.add.assert_called_once()

    def test_traced_decorator_metric_factory_error(self):
        """Test error handling when metric factory fails."""
        def failing_factory(config):
            raise RuntimeError("factory error")

        @traced(counter="calls", counter_factory=failing_factory)
        def func_with_bad_factory():
            return "result"

        # The function should still execute, but metric recording should fail gracefully
        with pytest.raises(RuntimeError):
            func_with_bad_factory()


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_traced_decorator_with_none_counter_factory(self):
        """Test @traced when counter_factory is None."""
        @traced(counter="calls", counter_factory=None)
        def func_no_factory():
            return "result"

        result = func_no_factory()
        assert result == "result"

    def test_traced_decorator_with_class_method(self):
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
    async def test_traced_decorator_async_method(self):
        """Test @traced decorator on async methods."""
        class MyClass:
            @traced()
            async def async_method(self):
                await asyncio.sleep(0.001)
                return "async result"

        obj = MyClass()
        result = await obj.async_method()
        assert result == "async result"

    def test_traced_decorator_nested_calls(self):
        """Test @traced decorator with nested function calls."""
        @traced()
        def outer_func():
            return inner_func()

        @traced()
        def inner_func():
            return "inner result"

        result = outer_func()
        assert result == "inner result"

    def test_traced_decorator_with_generator(self):
        """Test @traced decorator with generator function."""
        @traced()
        def generator_func():
            yield 1
            yield 2
            yield 3

        gen = generator_func()
        values = list(gen)
        assert values == [1, 2, 3]

    def test_traced_decorator_with_none_return(self):
        """Test @traced decorator with function that returns None."""
        mock_factory = MagicMock()
        mock_counter = MagicMock()
        mock_factory.return_value = mock_counter

        @traced(counter="calls", counter_factory=mock_factory, label_fn=lambda r, e, **kw: {})
        def returns_none():
            return None

        result = returns_none()
        assert result is None
        mock_counter.add.assert_called_once()
