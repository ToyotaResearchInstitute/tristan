import logging
from functools import lru_cache


# Keep track of 100 different messages and then error again
@lru_cache(1000)
def log_error_once(msg: str):
    """Log once during the lifetime of the program.
    Obviously it will log more if there's more than 100 unique calls, but that we can live with.
    """
    logging.error(msg)


# Keep track of 100 different messages and then warn again
@lru_cache(1000)
def log_warning_once(msg: str):
    """Log once during the lifetime of the program.
    Obviously it will log more if there's more than 100 unique calls, but that we can live with.
    """
    logging.warning(msg)
