#!/usr/bin/env bash
# Tail any training log live. Defaults to the most recent run under runs/.
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -ge 1 ]]; then
  LOG="$1"
else
  LOG=$(ls -t runs/*/log.txt runs/*_log.txt 2>/dev/null | head -n 1)
fi

if [[ -z "${LOG:-}" || ! -f "$LOG" ]]; then
  echo "no log found"; exit 1
fi

echo "tailing $LOG"
tail -f -n +1 "$LOG"
