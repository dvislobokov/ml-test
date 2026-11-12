#!/bin/bash
# Эксперимент 7, вторая часть: качество одного шага (равное число шагов, расписание по шагам).
cd "$(dirname "$0")"
r() { python3 track_exp.py "$@" 2>&1 | grep -E "RESULT|Error|Traceback"; }
for seed in 0 1; do
  r muon  8 0.03 1200st $seed
  r track 8 0.03 1200st $seed 1.0 0 1 4
  r track 8 0.03 1200st $seed 1.0 0 1 2
  r muon  32 0.03 500st $seed
  r track 32 0.03 500st $seed 1.0 0 1 4
  r track 32 0.03 500st $seed 1.0 0 1 2
done
echo ALL_DONE
