#!/bin/bash
# Эксперимент 7: гибрид «пересчёт раз в K шагов + слежение» против Muon, равный бюджет 45 с.
cd "$(dirname "$0")"
r() { python3 track_exp.py "$@" 2>&1 | grep -E "RESULT|Error|Traceback"; }
for seed in 0 1 2; do
  for b in 8 16 32; do
    r muon  $b 0.03 45 $seed
    r track $b 0.03 45 $seed 1.0 0 1 4
  done
done
for seed in 0 1; do
  for b in 8 16 32; do
    r muon3 $b 0.03 45 $seed
    r track $b 0.03 45 $seed 1.0 0 1 2
  done
done
echo ALL_DONE
