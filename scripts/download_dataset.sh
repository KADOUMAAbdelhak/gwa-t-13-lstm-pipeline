#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="data/raw"

echo "Verifying that the Materna traces are available locally under ${DATA_ROOT} ..."

MISSING=0
for TRACE in Materna-Trace-1 Materna-Trace-2 Materna-Trace-3; do
  if [[ ! -d "${DATA_ROOT}/${TRACE}" ]]; then
    echo "  - ${TRACE} is missing."
    MISSING=1
  else
    COUNT=$(find "${DATA_ROOT}/${TRACE}" -maxdepth 1 -name "*.csv" | wc -l | tr -d ' ')
    echo "  - ${TRACE} present (${COUNT} CSV files detected)."
  fi
done

if [[ "${MISSING}" -eq 1 ]]; then
  cat <<EOF

One or more trace directories are missing. Please place the GWA-T-13 Materna
dataset under ${DATA_ROOT}/Materna-Trace-{1,2,3} and re-run this check.
The dataset is intentionally distributed offline; there is nothing to download.
EOF
  exit 1
fi

echo "All traces detected. No download was performed (dataset is provided locally)."