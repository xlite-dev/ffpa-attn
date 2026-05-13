"""Logging utilities for FFPA.

This module provides a lightweight package logger used by command-line tools
and runtime debug paths.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import torch.distributed as dist

LOGGER_LEVEL_ENV_VAR = "FFPA_LOGGER_LEVEL"
RANK0_LOGGING_ENV_VAR = "FFPA_FORCE_ONLY_RANK0_LOGGING"

_FORMAT = "[%(asctime)s] [FFPA] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

_root_logger = logging.getLogger("FFPA")
_default_handler: logging.Handler | None = None
_log_once_messages: set[tuple[str, int, str]] = set()


def _log_level_from_env() -> int:
  """Return the configured FFPA log level.

  :return: Numeric logging level derived from ``FFPA_LOGGER_LEVEL``.
  """
  level_name = os.environ.get(LOGGER_LEVEL_ENV_VAR, "INFO").upper()
  level = getattr(logging, level_name, logging.INFO)
  return level if isinstance(level, int) else logging.INFO


def _truthy_env(name: str) -> bool:
  """Return whether an environment variable is enabled.

  :param name: Environment variable name.
  :return: ``True`` for common truthy values, otherwise ``False``.
  """
  return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


class NewLineFormatter(logging.Formatter):
  """Add the logging prefix to continuation lines."""

  def format(self, record: logging.LogRecord) -> str:
    """Format one log record.

    :param record: Log record emitted by ``logging``.
    :return: Formatted message with aligned multiline prefixes.
    """
    message = super().format(record)
    if record.message:
      prefix = message.split(record.message, maxsplit=1)[0]
      message = message.replace("\n", "\r\n" + prefix)
    return message


class Rank0Filter(logging.Filter):
  """Allow logs only from rank 0 when explicitly requested."""

  def filter(self, record: logging.LogRecord) -> bool:
    """Return whether the record should be emitted.

    :param record: Log record emitted by ``logging``.
    :return: ``True`` when the record is visible for the current rank.
    """
    del record
    if not _truthy_env(RANK0_LOGGING_ENV_VAR):
      return True
    return not (dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)


def _setup_logger() -> None:
  """Configure the package-level FFPA logger."""
  global _default_handler

  level = _log_level_from_env()
  _root_logger.setLevel(level)
  _root_logger.propagate = False
  if _default_handler is None:
    for handler in _root_logger.handlers:
      if getattr(handler, "_ffpa_default_handler", False):
        _default_handler = handler
        break
  if _default_handler is None:
    _default_handler = logging.StreamHandler(sys.stdout)
    setattr(_default_handler, "_ffpa_default_handler", True)
    _root_logger.addHandler(_default_handler)
  _default_handler.setFormatter(NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT))
  if not any(isinstance(item, Rank0Filter) for item in _default_handler.filters):
    _default_handler.addFilter(Rank0Filter())
  _default_handler.setLevel(level)


def _render_message(logger: logging.Logger, level: int, msg: object, args: tuple[Any, ...]) -> str:
  """Render a logging message with standard ``%`` interpolation.

  :param logger: Logger receiving the message.
  :param level: Numeric logging level.
  :param msg: Message template.
  :param args: Positional formatting arguments.
  :return: Rendered message used for once de-duplication.
  """
  return logging.LogRecord(
    name=logger.name,
    level=level,
    pathname="",
    lineno=0,
    msg=msg,
    args=args,
    exc_info=None,
  ).getMessage()


def _log_once(logger: logging.Logger, level: int, msg: object, *args: Any, **kwargs: Any) -> None:
  """Emit one log message only once per logger, level, and rendered text.

  :param logger: Logger receiving the message.
  :param level: Numeric logging level.
  :param msg: Message template.
  :param args: Positional formatting arguments.
  :param kwargs: Keyword arguments forwarded to ``Logger.log``.
  """
  message = _render_message(logger, level, msg, args)
  key = (logger.name, level, message)
  if key in _log_once_messages:
    return
  _log_once_messages.add(key)
  logger.log(level, msg, *args, **kwargs)


def _info_once(self: logging.Logger, msg: object, *args: Any, **kwargs: Any) -> None:
  _log_once(self, logging.INFO, msg, *args, **kwargs)


def _debug_once(self: logging.Logger, msg: object, *args: Any, **kwargs: Any) -> None:
  _log_once(self, logging.DEBUG, msg, *args, **kwargs)


def _warning_once(self: logging.Logger, msg: object, *args: Any, **kwargs: Any) -> None:
  _log_once(self, logging.WARNING, msg, *args, **kwargs)


logging.Logger.info_once = _info_once  # type: ignore[attr-defined]
logging.Logger.debug_once = _debug_once  # type: ignore[attr-defined]
logging.Logger.warning_once = _warning_once  # type: ignore[attr-defined]


def init_logger(name: str) -> logging.Logger:
  """Initialize a logger with FFPA's default console handler.

  :param name: Logger name, usually ``__name__`` from the caller module.
  :return: Configured logger instance.
  """
  _setup_logger()
  level = _log_level_from_env()
  logger = logging.getLogger(name)
  logger.setLevel(level)
  logger.propagate = False
  if _default_handler is not None and _default_handler not in logger.handlers:
    logger.addHandler(_default_handler)
  return logger


_setup_logger()

__all__ = [
  "LOGGER_LEVEL_ENV_VAR",
  "RANK0_LOGGING_ENV_VAR",
  "NewLineFormatter",
  "Rank0Filter",
  "init_logger",
]
