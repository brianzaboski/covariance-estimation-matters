#!/usr/bin/env bash
# Run every analysis script in order, then the two checks.
#
#   ./run_all.sh              full run, needs the registry data in data/
#   ./run_all.sh synthetic    Study 1 and the three Study 2 scripts against the
#                             synthetic file; 01, 02 and 02b are skipped, since
#                             they need the item-level export
#
# Stops at the first failure rather than carrying on with missing inputs.

set -euo pipefail
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "No .venv found. Using whatever python is on PATH."
    echo "Create one with:  python -m venv .venv && source .venv/bin/activate && pip install -r env/requirements.txt"
    echo
fi

PY=$(command -v python3 || command -v python)

# Show which interpreter is actually being used, so a wrong environment is
# visible before the analysis scripts run against it.
"$PY" -c "import sys; print('python', sys.version.split()[0]); print('from   ', sys.executable)"

"$PY" - <<'EOF' || echo "
WARNING: scikit-learn is not 1.8.0. MinCovDet changed across releases, so every
MCD number will differ from the manuscript. Continuing anyway.
"
import sklearn, sys
print("scikit-learn", sklearn.__version__)
sys.exit(0 if sklearn.__version__ == "1.8.0" else 1)
EOF

DATA=()
if [ "${1:-}" = "synthetic" ]; then
    DATA=(--data data/synthetic/synthetic_clinical_data.csv)
    echo "Running against the synthetic dataset. Numbers will not match the paper."
    echo
fi

run () {
    echo
    echo "=== $1 ==="
    if ! "$PY" "code/$1" "${@:2}"; then
        echo
        echo "FAILED: $1 stopped with an error. Fix it before continuing;"
        echo "later scripts read what the earlier ones write."
        exit 1
    fi
}

if [ "${1:-}" != "synthetic" ]; then
    run 01_clean_and_score.py
    run 02_descriptives_octcdq_bfas.py
    run 02b_feature_reliability.py
fi

run 03_phase2_analyses.py "${DATA[@]}"
run 04_invert_r.py
run 05_study1_simulation.py
run 05b_figure1.py
run 06_bootstrap_sample_size.py "${DATA[@]}"
run 07_support_fraction_sweep.py "${DATA[@]}"
run 08_fig_mcd_discarded.py

echo
echo "=== Checks ==="
"$PY" code/check_reproduction.py
echo
"$PY" code/check_outputs.py --quiet
echo
echo "Done. Read the two check reports above."
