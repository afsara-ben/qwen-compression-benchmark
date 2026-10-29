#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if ! command -v python3 >/dev/null 2>&1; then
  echo 'Python 3.10+ is required for bootstrap. The launcher installs the pinned Python 3.12 runtime.' >&2
  exit 2
fi
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10+ is required for bootstrap")'
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m qbench.cli "$@"
