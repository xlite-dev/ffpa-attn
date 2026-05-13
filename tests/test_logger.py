"""Unit tests for FFPA logging helpers."""

import importlib
import io
import logging

from ffpa_attn import logger as ffpa_logger


def test_init_logger_respects_ffpa_logger_level(monkeypatch):
  monkeypatch.setenv(ffpa_logger.LOGGER_LEVEL_ENV_VAR, "DEBUG")
  reloaded_logger = importlib.reload(ffpa_logger)

  try:
    logger = reloaded_logger.init_logger("ffpa_attn.tests.debug_level")

    assert logger.level == logging.DEBUG
  finally:
    monkeypatch.setenv(ffpa_logger.LOGGER_LEVEL_ENV_VAR, "INFO")
    importlib.reload(ffpa_logger)


def test_once_logging_dedupes_by_logger_level_and_message():
  ffpa_logger._log_once_messages.clear()
  stream = io.StringIO()
  logger = logging.getLogger("ffpa_attn.tests.once")
  logger.handlers.clear()
  logger.setLevel(logging.DEBUG)
  logger.propagate = False
  handler = logging.StreamHandler(stream)
  handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
  logger.addHandler(handler)

  try:
    logger.info_once("same %s", "message")
    logger.info_once("same %s", "message")
    logger.debug_once("same %s", "message")
    logger.debug_once("same %s", "message")
    logger.warning_once("same %s", "message")
    logger.warning_once("different message")
  finally:
    logger.handlers.clear()
    ffpa_logger._log_once_messages.clear()

  assert stream.getvalue().splitlines() == [
    "INFO:same message",
    "DEBUG:same message",
    "WARNING:same message",
    "WARNING:different message",
  ]
