import logging
from pythonjsonlogger.json import JsonFormatter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry._logs import set_logger_provider, get_logger_provider
from opentelemetry.instrumentation.logging import LoggingInstrumentor

LoggingInstrumentor().instrument()


def configure_logging(
    server,
    resource,
    log_level=logging.NOTSET,
):
    otlp_log_exporter = OTLPLogExporter(endpoint="http://{}/v1/logs".format(server))

    # Set up the logger provider with a batch log processor
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(otlp_log_exporter))
    set_logger_provider(logger_provider)

    # Set up Python logging integration
    provider = get_logger_provider()
    logger = logging.getLogger()
    if not any(isinstance(h, LoggingHandler) for h in logging.getLogger().handlers):
        handler = LoggingHandler(level=log_level, logger_provider=provider)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(log_level)
