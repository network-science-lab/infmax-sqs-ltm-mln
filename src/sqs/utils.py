"""Shared utilities for the sqs package."""

import datetime
import logging

import git

logger = logging.getLogger("sqs")


def get_current_time() -> str:
    """Return the current wall-clock time formatted as ``YYYY-MM-DD HH:MM:SS``."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_recent_git_sha() -> str:
    """Return the HEAD commit SHA of the repository containing this file."""
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha
