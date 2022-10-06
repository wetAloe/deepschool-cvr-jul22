import logging


def init_logging():
    logging_format = (
        '[{asctime}] {levelname} {filename}:{lineno} {funcName} {message}'
    )
    formatter = logging.Formatter(logging_format, style='{')

    logger = logging.getLogger()
    logger.handlers = []

    logger.setLevel('INFO')

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
