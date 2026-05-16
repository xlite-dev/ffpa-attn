# Runtime-only helpers used by production entrypoints.
#
# Keep this module free of test/reference dependencies so importing the public
# SplitD API does not require packages such as einops.

from torch._guards import active_fake_mode


def is_fake_mode() -> bool:
  return active_fake_mode() is not None
