#!/usr/bin/env bash
# Rebuild the designated final submission (det_ens_AD) from scratch and compare it byte-for-byte
# with the file that was uploaded to Kaggle. Run from midterm/ after src/prep.py.
#   det_ens_AD = within-user rank average of four LightGBM LambdaRank ensembles (5 seeds each):
#   {screen time summed over sessions, longest single session} x {test history 15-19+28-30, 20-24+28-30}
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -I"
for spec in "sum clean8" "max clean8" "sum recent8" "max recent8"; do
  read -r variant hist <<< "$spec"
  $PY src/predict_test.py --name "repro_${variant}_${hist}" --ensemble lr31_r250 --seeds 5 \
      --variant "$variant" --test-hist "$hist" > /dev/null 2>&1
done
$PY src/blend_submission.py --out repro_ens_AD \
    --parts repro_sum_clean8 repro_max_clean8 repro_sum_recent8 repro_max_recent8 > /dev/null 2>&1
new=$(shasum -a 256 submissions/repro_ens_AD.csv | cut -d' ' -f1)
old=$(shasum -a 256 submissions/det_ens_AD.csv | cut -d' ' -f1)
mkdir -p cache/repro && mv submissions/repro_*.csv cache/repro/ && mv evidence/repro_*_info.json cache/repro/
printf '{\n  "sha256_submitted": "%s",\n  "sha256_rebuilt": "%s",\n  "identical": %s\n}\n' \
    "$old" "$new" "$([ "$new" = "$old" ] && echo true || echo false)" | tee evidence/reproduce_final.json
