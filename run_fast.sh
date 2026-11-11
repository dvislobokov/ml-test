#!/bin/bash
# Эксперимент 3 (дешёвая ортогонализация) и эксперимент 4 (bfloat16 для всей модели).
cd "$(dirname "$0")"
f() { python3 fast_muon_exp.py "$@" 2>&1 | grep -E "RESULT|Error|Traceback"; echo "DONE fast $*"; }
m() { AMP=$1 python3 muon_exp.py "${@:2}" 2>&1 | grep -E "Error|Traceback"; echo "DONE amp=$1 ${*:2}"; }
for seed in 0 1; do
  for v in cold-5 cold-3 cold-2 lazy-4 lazy-16 warm; do f mlp $v 0.01 1000 $seed; done
done
for v in cold-5 cold-3 cold-2 lazy-4 warm; do f gpt $v 0.03 500 0; done
m 1 mlp adamw 1e-3 1000 0
m 0 mlp adamw 1e-3 1000 1
m 1 mlp adamw 1e-3 1000 1
m 1 gpt adamw 1e-3 500 0
m 1 gpt muon 0.03 500 0
echo ALL_DONE
