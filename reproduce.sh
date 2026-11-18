#!/usr/bin/env bash
# Reproduce every experiment of the replication of
# "An axiomatic foundation for optimized certainty equivalents"
# (Blier-Wong, Lauzier and Van Oosten, 2026).
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== Running tests =="
python -m pytest || exit 1

for n in 1 2 3 4 5 6 7; do
    echo "== Running experiment exp_${n} =="
    python -m "exp.exp_${n}" || exit 1
done

echo "== Done. See results/RESULTS.md for the comparison against the paper. =="
