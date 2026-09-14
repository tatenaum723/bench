#!/usr/bin/env bash
# Breadth run: every checkpoint x {nothink, think} x 3 samples, vendor sampling.
# Base models first; quant variants are gated on bringup.py passing (parsers +
# long context on real weights). Resumable: rerun this script to continue.
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=results/_queue.log
mkdir -p results
stamp() { date "+%F %T"; }

for model in granite42-8b gemma4-12b mellum2-12b qwen3.5-9b; do
  echo "$(stamp) RUN $model" | tee -a $LOG
  $PY -u run.py "$model" --samples 3 >> "results/_run_${model}.log" 2>&1
  echo "$(stamp) DONE $model (exit $?)" | tee -a $LOG
done

for model in gemma4-12b-nvfp4 gemma4-12b-awq mellum2-12b-awq granite42-8b-fp8; do
  echo "$(stamp) BRINGUP $model" | tee -a $LOG
  $PY -u bringup.py "$model" --long-tokens $([ "$model" = granite42-8b-fp8 ] && echo 14000 || echo 28000) \
      >> "results/_bringup_${model}.log" 2>&1
  if $PY - "$model" <<'PYEOF'
import json, sys
r = json.load(open(f"results/_bringup/{sys.argv[1]}.json"))
fatal = r.get("fatal")
crit = [c for c in r["checks"] if not c["ok"] and ("parsed" in c["name"] or "survived" in c["name"])]
sys.exit(1 if fatal or crit else 0)
PYEOF
  then
    echo "$(stamp) RUN $model" | tee -a $LOG
    $PY -u run.py "$model" --samples 3 >> "results/_run_${model}.log" 2>&1
    echo "$(stamp) DONE $model (exit $?)" | tee -a $LOG
  else
    echo "$(stamp) SKIP $model: bring-up failed (results/_bringup/${model}.json)" | tee -a $LOG
  fi
done
echo "$(stamp) QUEUE COMPLETE" | tee -a $LOG
