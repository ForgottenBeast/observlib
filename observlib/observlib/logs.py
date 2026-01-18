import logging
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry._logs import set_logger_provider, get_logger_provider
from opentelemetry.instrumentation.logging import LoggingInstrumentor

logger = logging.getLogger(__name__)


def _has_handler(handlers, handler_type):
    """Check if a handler of the given type is already present.

    Args:
        handlers: List of handlers to check
        handler_type: The handler class type to look for

    Returns:
        True if a handler of the specified type exists, False otherwise
    """
    return any(isinstance(h, handler_type) for h in handlers)


def configure_logging(
    server,
    resource,
    log_level=logging.NOTSET,
):
    try:
        # Instrument Python logging to capture logs with OpenTelemetry
        LoggingInstrumentor().instrument(set_logging_format=True)

        root_logger = logging.getLogger()

        if server:
            # Configure OTLP log exporter
            otlp_log_exporter = OTLPLogExporter(endpoint=f"http://{server}/v1/logs")

            # Set up the logger provider with a batch log processor
            logger_provider = LoggerProvider(resource=resource)
            logger_provider.add_log_record_processor(BatchLogRecordProcessor(otlp_log_exporter))
            set_logger_provider(logger_provider)

            # Set up Python logging integration
            provider = get_logger_provider()
            if not _has_handler(root_logger.handlers, LoggingHandler):
                handler = LoggingHandler(level=log_level, logger_provider=provider)
                root_logger.addHandler(handler)
                root_logger.setLevel(log_level)
        else:
            # No server configured - set up basic logging to stdout
            if not _has_handler(root_logger.handlers, logging.StreamHandler):
                stdout_handler = logging.StreamHandler()
                stdout_handler.setLevel(log_level)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                stdout_handler.setFormatter(formatter)
                root_logger.addHandler(stdout_handler)
            root_logger.setLevel(log_level)
    except Exception as e:
        logger.error(f"Failed to configure logging: {e}")
        raise
