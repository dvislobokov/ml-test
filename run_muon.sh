#!/bin/bash
# Подбор шага (короткие прогоны) + финальные длинные прогоны лучших конфигураций.
cd "$(dirname "$0")"
run() { python3 muon_exp.py "$@" 2>&1 | grep -E "^(gpt|mlp)"; echo "DONE $*"; }
for c in "muon 0.01" "muon 0.03" "muon 0.1"; do run mlp $c 1000 0; done
for c in "adamw 3e-4" "adamw 2e-3"; do run gpt $c 500 0; done
for seed in 1 2; do
  run gpt adamw 1e-3 1500 $seed
  run gpt muon 0.03 1500 $seed
done
echo ALL_DONE
