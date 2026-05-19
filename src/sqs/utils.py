"""Shared utilities for the sqs package."""

import datetime
import logging

import git

logger = logging.getLogger("sqs")


def initialise_logger(level: int = logging.INFO) -> None:
    """
    Attach a StreamHandler with a standard formatter to the sqs logger.

    Intended to be called once from the CLI entrypoint.

    :param level: logging level, e.g. ``logging.DEBUG`` or ``logging.INFO``.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)


def get_current_time() -> str:
    """Return the current wall-clock time formatted as ``YYYY-MM-DD HH:MM:SS``."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_recent_git_sha() -> str:
    """Return the HEAD commit SHA of the repository containing this file."""
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha
