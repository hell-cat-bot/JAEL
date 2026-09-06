#!/usr/bin/env bash
# Builds the zip to hand to a code reviewer (Claude Code or human).
#
# Excludes generated data (regenerable in ~6 s via scripts/run_v1.py), bytecode,
# and caches. Keeps every source file, all four documents, the notebook, and the
# measured results JSON -- which is what a reviewer actually needs.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-jale_review_bundle.zip}"
rm -f "$OUT"

zip -qr "$OUT" jale \
  -x 'jale/data/*' \
  -x '*/__pycache__/*' \
  -x '*.pyc' \
  -x 'jale/artifacts/*' \
  -x '*.ipynb_checkpoints/*'

echo "built $OUT"
echo
echo "contents:"
unzip -l "$OUT" | tail -n +4 | head -n -2 | awk '{print "  "$4"  ("$1" bytes)"}'
echo
echo "size: $(du -h "$OUT" | cut -f1)"
