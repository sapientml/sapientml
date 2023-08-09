import logging
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(scope="session", autouse=True)
def disable_logging():
    logging.disable(logging.FATAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture(scope="function", autouse=True)
def reset_sapientml_logger():
    # FIXME: more efficient way to reset a logger
    logger = logging.getLogger("sapientml")
    logger.handlers.clear()
    logger.root.handlers.clear()


@pytest.fixture(scope="function", autouse=True)
def path_home(tmp_path):
    with mock.patch.object(Path, "home"):
        yield Path(tmp_path)
