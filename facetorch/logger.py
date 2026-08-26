import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Optional

from facetorch.exceptions import ConfigurationError

try:
    from pythonjsonlogger.json import JsonFormatter
except ImportError:  # pragma: no cover - compatibility with python-json-logger<4
    from pythonjsonlogger import jsonlogger

    JsonFormatter = jsonlogger.JsonFormatter


class CustomJsonFormatter(JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        # Remove taskName field if it exists and is None
        if "taskName" in log_record and log_record["taskName"] is None:
            del log_record["taskName"]


def get_logger(name: str = "facetorch") -> logging.Logger:
    """Return a shared logger without changing its configured state.

    Module-level timing hooks use this accessor so importing a component cannot
    override logging that an application configured explicitly.
    """

    return logging.getLogger(name)


class LoggerJsonFile:
    def __init__(
        self,
        name: str = "facetorch",
        level: int = logging.CRITICAL,
        path_file: Optional[str] = None,
        json_format: str = "%(asctime)s %(levelname)s %(message)s",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 3,
    ):
        """Logger in json format that writes to a file and console.

        Args:
            name (str): Name of the logger.
            level (str): Level of the logger.
            path_file (str): Path to the log file.
            json_format (str): Format of the log record.
            max_bytes (int): Rotate file logs after this many bytes.
            backup_count (int): Number of rotated log files to retain.

        Attributes:
            logger (logging.Logger): Logger object.

        """
        self.name = name
        self.level = level
        self.path_file = path_file
        self.json_format = json_format
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ConfigurationError("max_bytes must be a positive integer.")
        if (
            isinstance(backup_count, bool)
            or not isinstance(backup_count, int)
            or backup_count < 1
        ):
            raise ConfigurationError("backup_count must be a positive integer.")
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        self.logger = logging.getLogger(self.name)
        self.configure()

    def configure(self):
        """Configure JSON console/file handlers idempotently."""
        self.logger.setLevel(self.level)

        formatter = CustomJsonFormatter(fmt=self.json_format)
        stream_handlers = [
            handler
            for handler in self.logger.handlers
            if getattr(handler, "_facetorch_stream_handler", False)
        ]
        if not stream_handlers:
            json_handler = logging.StreamHandler()
            json_handler._facetorch_stream_handler = True
            self.logger.addHandler(json_handler)
            stream_handlers = [json_handler]
        for handler in stream_handlers:
            handler.setLevel(self.level)
            handler.setFormatter(formatter)

        normalized_path = (
            os.path.abspath(os.fspath(self.path_file))
            if self.path_file is not None
            else None
        )
        managed_file_handlers = [
            handler
            for handler in self.logger.handlers
            if getattr(handler, "_facetorch_file_handler", False)
        ]
        for handler in managed_file_handlers:
            if normalized_path is None or handler.baseFilename != normalized_path:
                self.logger.removeHandler(handler)
                handler.close()

        if normalized_path is not None:
            parent = os.path.dirname(normalized_path)
            if parent:
                try:
                    os.makedirs(parent, exist_ok=True)
                except OSError as exc:
                    raise ConfigurationError(
                        f"Cannot create log directory {parent!r}. Choose a writable "
                        "path_file or leave file logging disabled."
                    ) from exc
            file_handlers = [
                handler
                for handler in self.logger.handlers
                if getattr(handler, "_facetorch_file_handler", False)
                and handler.baseFilename == normalized_path
            ]
            for handler in file_handlers[1:]:
                self.logger.removeHandler(handler)
                handler.close()
            file_handlers = file_handlers[:1]
            if not file_handlers:
                path_file_handler = RotatingFileHandler(
                    normalized_path,
                    mode="a",
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count,
                )
                path_file_handler._facetorch_file_handler = True
                path_file_handler.setLevel(self.level)
                path_file_handler.setFormatter(formatter)
                self.logger.addHandler(path_file_handler)
            else:
                for handler in file_handlers:
                    handler.maxBytes = self.max_bytes
                    handler.backupCount = self.backup_count
                    handler.setLevel(self.level)
                    handler.setFormatter(formatter)

        self.logger.propagate = False
