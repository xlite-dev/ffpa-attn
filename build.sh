#!/usr/bin/env bash
# Convenience wrapper that forwards all arguments to tools/build_fast.sh.
# Usage: bash build.sh --arch sm_120f --ext all --headdim all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/tools/build_fast.sh" "$@"
