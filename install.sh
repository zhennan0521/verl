#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Installing peft (editable) ==="
pip install -e "$SCRIPT_DIR/peft" --no-deps --no-build-isolation

echo "=== Installing verl (editable) ==="
pip install -e "$SCRIPT_DIR" --no-deps --no-build-isolation

echo "=== Verifying peft installation ==="
PEFT_LOCATION=$(python3 -c "import peft; print(peft.__file__)")
if echo "$PEFT_LOCATION" | grep -q "$SCRIPT_DIR/peft"; then
    echo "[OK] peft loaded from editable install: $PEFT_LOCATION"
else
    echo "[FAIL] peft NOT loaded from editable install!"
    echo "  Expected path containing: $SCRIPT_DIR/peft"
    echo "  Actual: $PEFT_LOCATION"
    exit 1
fi

echo "=== Summary ==="
pip show peft | grep -E "^(Name|Version|Location|Editable)"
pip show verl | grep -E "^(Name|Version|Location|Editable)"
