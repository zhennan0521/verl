#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Installing peft (editable) ==="
pip install -e "$SCRIPT_DIR/peft" --no-deps --no-build-isolation

echo "=== Installing verl (editable) ==="
pip install -e "$SCRIPT_DIR" --no-deps --no-build-isolation

echo "=== Done ==="
pip show peft | grep -E "^(Name|Version|Location)"
pip show verl | grep -E "^(Name|Version|Location)"
