#!/bin/bash
# Эксперимент 5: прореженный обратный проход и контроль «меньший батч», равный бюджет времени.
cd "$(dirname "$0")"
for seed in 0 1; do
  for opt in adamw muon; do
    for v in "32 1.0" "32 0.25" "32 0.125" "16 1.0" "8 1.0"; do
      python3 sampled_exp.py $opt $v 45 $seed 2>&1 | grep -E "RESULT|Error|Traceback"
    done
  done
done
echo ALL_DONE
