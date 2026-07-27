import logging

try:
    import loguru
except ModuleNotFoundError:
    loguru = None


def _get_logger():
    if loguru is not None:
        return loguru.logger

    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("quant_videogen")


logger = _get_logger()
