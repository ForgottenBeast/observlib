# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`observlib` is a Python library that provides a unified interface for observability (tracing, metrics, and logging) using OpenTelemetry. It simplifies configuration of distributed tracing, metric collection, and structured logging with support for OTLP, Prometheus, and Pyroscope backends.

## Directory Structure

```
observlib/
├── observlib/          # Main package
│   ├── __init__.py     # Main API: configure_telemetry()
│   ├── decorator.py    # @traced decorator for instrumenting functions
│   ├── logs.py         # Logging configuration with OTLP
│   ├── metrics.py      # Metrics configuration (Prometheus + OTLP)
│   └── traces.py       # Tracing configuration with Pyroscope support
├── tests/              # Test suite
│   └── test_observlib.py  # Integration tests using pytest
├── nix/               # Nix development environment
├── pyproject.toml     # Project metadata and dependencies
├── pytest.ini         # Pytest configuration
└── uv.lock            # Locked dependency versions
```

## Core Architecture

The library has three main configuration modules that work together:

1. **Tracing** (`traces.py`): Sets up OpenTelemetry TracerProvider with OTLP span exporter and Pyroscope span processor
2. **Metrics** (`metrics.py`): Configures MeterProvider with Prometheus reader and optional OTLP exporter
3. **Logging** (`logs.py`): Sets up LoggerProvider with OTLP log exporter and Python logging integration

The `@traced` decorator (`decorator.py`) is the main user-facing API that:
- Creates spans for function execution
- Records execution time as histogram metrics
- Records function calls as counter metrics
- Supports both sync and async functions
- Allows custom labels and amounts via `label_fn` and `amount_fn` callbacks

The `configure_telemetry()` function in `__init__.py` is the entry point that:
- Sets up all three telemetry components
- Accepts optional OTLP server endpoint, Pyroscope server, and resource attributes
- Creates a Resource with service.name and any custom attributes

## Development Setup

The project uses Nix for reproducible development environments:

```bash
# Enter development shell
nix flake update  # Update dependencies if needed
nix develop       # Activates development environment with all dependencies
```

All dependencies (both main and test) are installed via the Nix devshell. When you add new dependencies to `pyproject.toml`, update `nix/shells/dev/default.nix` to include them in the `packages` list.

## Common Commands

**Code Quality:**
```bash
ruff check observlib/        # Lint code
ruff format observlib/       # Format code
```

**Running Code:**
```bash
cd observlib
python -c "from observlib import configure_telemetry; ..."  # Direct imports
```

**Testing:**
```bash
cd observlib
# Install test dependencies
nix develop --command uv sync --extra test

# Run all tests
nix develop --command uv run pytest -v

# Run specific test file
nix develop --command uv run pytest tests/test_observlib.py -v
```

**Dependency Management:**
1. Update `pyproject.toml` to add/remove dependencies
2. Update `nix/shells/dev/default.nix` to include the new packages in the `packages` list with `python313Packages.package_name`
3. Re-enter the nix shell: `exit` and `nix develop`

## Key Dependencies

- **OpenTelemetry**: Distributed tracing and metrics instrumentation
  - `opentelemetry-api`: Core API
  - `opentelemetry-sdk`: SDK with exporters and processors
  - `opentelemetry-exporter-otlp`: HTTP-based OTLP protocol exporter
  - `opentelemetry-exporter-prometheus`: Prometheus metrics reader
  - `opentelemetry-instrumentation-logging`: Python logging integration

- **Pyroscope**: Continuous profiling
  - `pyroscope-io`: Profiler client
  - `pyroscope-otel`: OpenTelemetry integration

## Important Implementation Details

### The @traced Decorator

The decorator in `decorator.py:8` wraps both sync and async functions. Key features:

- **Tracing**: Always creates a span using the tracer provider (line 111/148)
- **Metrics Recording**:
  - Histograms record execution time (line 51-73)
  - Counters track invocations with configurable amounts (line 88-104)
- **Error Handling**: Sets span status to ERROR on exceptions (line 119/157)
- **Flexible Configuration**:
  - `timer`/`timer_factory`: Configure histogram metrics
  - `counter`/`counter_factory`: Configure counter metrics
  - `label_fn`: Function returning dict of custom labels
  - `amount_fn`: Function returning amount to increment counter
  - `tracer`: Specify tracer instance (defaults to global tracer)
  - `func_name_as_label`: Automatically adds function name to labels (default: True)

### Custom Resource Attributes

The `configure_telemetry()` function accepts `resource_attrs={}` (line 18 in `__init__.py`) which are merged with the service.name attribute using dictionary union operator (line 28).

### Configuration Flow

1. Pyroscope configured if `pyroscope_server` provided (line 20-25)
2. Resource created with service.name + custom attrs (line 27-29)
3. Tracing configured if `server` provided (line 32)
4. Logging configured if `server` provided (line 33)
5. Metrics always configured (line 35)

## Testing Approach

The library includes comprehensive integration tests in `observlib/tests/test_observlib.py` using pytest. The test suite covers:

- **configure_telemetry() tests**: Validation of service name requirements, configuration with/without OTLP server, resource attributes, Pyroscope integration, and logging setup
- **@traced decorator tests**:
  - Basic functionality with sync and async functions
  - Exception handling and error propagation
  - Metrics recording (histograms and counters)
  - Custom label and amount functions
  - Function signature preservation
  - Class methods and nested calls
  - Generator functions
- **Integration tests**: Real provider interactions and end-to-end flows

### Running Tests

Tests use pytest with the `pytest-asyncio` plugin for async test support. From the `observlib/` directory:

```bash
# Run all tests
nix develop --command uv run pytest -v

# Run with coverage
nix develop --command uv run pytest --cov=observlib

# Run specific test
nix develop --command uv run pytest tests/test_observlib.py::test_traced_decorator_async_function -v
```

### Test Dependencies

Test dependencies are defined in `pyproject.toml` under `[project.optional-dependencies]`:
- `pytest>=7.0`: Test framework
- `pytest-asyncio>=0.23.0`: Async test support
- `pytest-mock>=3.12.0`: Mocking utilities

Install with: `uv sync --extra test`
