import logging
from typing import List


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;5;244m"  # Gris
    white = "\x1b[38;5;255m"  # Blanc (plus visible)
    orange = "\x1b[38;5;214m"  # Orange
    red = "\x1b[31;20m"  # Rouge
    bold_red = "\x1b[31;1m"  # Rouge en gras
    green_violet = "\x1b[38;5;46m"  # Vert avec fond violet
    reset = "\x1b[0m"

    format_ = "%(asctime)s.%(msecs)03d  - %(name)s - %(message)s"

    logging.RESULT = 25

    FORMATS = {
        logging.DEBUG: grey + format_ + reset,
        logging.INFO: white + format_ + reset,
        logging.WARNING: orange + format_ + reset,
        logging.ERROR: red + format_ + reset,
        logging.CRITICAL: bold_red + format_ + reset,
        logging.RESULT: green_violet + format_ + reset,  # type: ignore
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


class Logger:
    logger = logging.getLogger("Spiderer")

    def __init__(self, reset: bool = False) -> None:
        if not reset:
            return
        Logger.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(CustomFormatter())

        def _result(logger, message, *args, **kwargs):
            if logger.isEnabledFor(logging.RESULT):  # type: ignore
                logger._log(logging.RESULT, message, args, **kwargs)  # type: ignore

        logging.RESULT = 25  # type: ignore
        logging.addLevelName(25, "RESULT")
        logging.Logger.result = _result  # type: ignore
        Logger.logger.addHandler(ch)

    @staticmethod
    def add_verbose() -> None:
        Logger.logger.setLevel(logging.DEBUG)

    @staticmethod
    def debug(*args: str) -> None:
        msg = " ".join(str(e) for e in args)
        Logger.logger.debug(msg)

    @staticmethod
    def info(*args: str):
        msg = " ".join(str(e) for e in args)
        Logger.logger.info(msg)

    @staticmethod
    def result(*args: str):
        msg = " ".join(str(e) for e in args)
        Logger.logger.result(msg)  # type: ignore

    @staticmethod
    def warning(*args: str):
        msg = " ".join(str(e) for e in args)
        Logger.logger.warning(msg)

    @staticmethod
    def error(*args: str):
        msg = " ".join(str(e) for e in args)
        Logger.logger.error(msg)


Logger(reset=True)
