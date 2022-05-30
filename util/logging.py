from contextlib import contextmanager
from loguru import logger
from tqdm import tqdm
import sys

@contextmanager
def logging_redirect_tqdm():
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    yield