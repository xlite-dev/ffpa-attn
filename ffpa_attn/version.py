"""Expose ``__version__`` resolved from setuptools-scm.

Priority:
  1. ``ffpa_attn/_version.py`` generated at build time by setuptools-scm.
  2. Installed distribution metadata (for pip-installed users).
  3. ``"0.0.0+unknown"`` as a last-resort fallback (e.g. editable checkout
     without a build).
"""

from __future__ import annotations

try:
  from ._version import version as __version__  # type: ignore[attr-defined]
except ImportError:
  try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
      __version__ = _pkg_version("ffpa-attn")
    except PackageNotFoundError:
      __version__ = "0.0.0+unknown"
  except ImportError:
    __version__ = "0.0.0+unknown"
