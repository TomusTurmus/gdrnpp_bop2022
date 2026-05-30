#!/usr/bin/env bash
set -eo pipefail

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  set +u
  conda activate gdrnpp || true
  set -u
fi

if [ "$#" -eq 0 ]; then
  exec /bin/bash
else
  exec "$@"
fi