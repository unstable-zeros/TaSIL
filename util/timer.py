from contextlib import contextmanager
from loguru import logger
import time

@contextmanager
def timed(name):
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    logger.info(f"{name} took {elapsed} seconds")