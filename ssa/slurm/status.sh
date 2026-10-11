#!/bin/bash
# Quick status of the experiment chain: queue, per-arm phase progress, latest evaluation numbers.
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
squeue -u $USER -o "%.10i %.12P %.28j %.2t %.10M %R" | grep -E "JOBID|ssa"
echo "--- arms: last finished phase / last eval ---"
for d in $SSA_ROOT/runs/*/; do
  arm=$(basename $d); [[ "$arm" == *_sft ]] && continue
  last=$(ls -d $d/phase_0*/final 2>/dev/null | sort | tail -1 | xargs -r dirname | xargs -r basename)
  ev=$(ls $d/eval_phase*.json 2>/dev/null | sort | tail -1)
  if [ -n "$ev" ]; then
    python3 - "$ev" <<'PY'
import json,sys; e=json.load(open(sys.argv[1])); a=(e.get("asa") or {})
print(f"  {sys.argv[1].split('/')[-2]:22s} {sys.argv[1].split('/')[-1]:20s} align={e.get('align_loss'):.3f} general={e.get('general_loss'):.3f} delta={e.get('delta_sum'):.2f} pref={e.get('pref_rate'):.3f} U={e.get('U_per_token'):.4f} beta={a.get('beta_s')}")
PY
  else echo "  $arm: no evaluation yet (last checkpoint: ${last:-none})"; fi
done
