import logging
import os
from typing import Optional

from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        # Remove taskName field if it exists and is None
        if "taskName" in log_record and log_record["taskName"] is None:
            del log_record["taskName"]


class LoggerJsonFile:
    def __init__(
        self,
        name: str = "facetorch",
        level: int = logging.CRITICAL,
        path_file: Optional[str] = None,
        json_format: str = "%(asctime)s %(levelname)s %(message)s",
    ):
        """Logger in json format that writes to a file and console.

        Args:
            name (str): Name of the logger.
            level (str): Level of the logger.
            path_file (str): Path to the log file.
            json_format (str): Format of the log record.

        Attributes:
            logger (logging.Logger): Logger object.

        """
        self.name = name
        self.level = level
        self.path_file = path_file
        self.json_format = json_format

        self.logger = logging.getLogger(self.name)
        self.configure()

    def configure(self):
        """Configures the logger."""
        if self.logger.level == 0 or self.level < self.logger.level:
            self.logger.setLevel(self.level)

        if len(self.logger.handlers) == 0:
            json_handler = logging.StreamHandler()
            formatter = CustomJsonFormatter(fmt=self.json_format)
            json_handler.setFormatter(formatter)
            self.logger.addHandler(json_handler)

            if self.path_file is not None:
                os.makedirs(os.path.dirname(self.path_file), exist_ok=True)
                path_file_handler = logging.FileHandler(self.path_file, mode="w")
                path_file_handler.setLevel(self.level)
                self.logger.addHandler(path_file_handler)

        self.logger.propagate = False
