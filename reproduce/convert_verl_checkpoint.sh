#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  convert_verl_all_steps_inplace.sh -i <exp_dir> [--python <python>] [--overwrite] [--backup]
                                   [--continue-on-error] [--steps 100,200,300] [--dry-run]

What it does (for each global_step_XXX):
  1) Merge FSDP -> HF into <exp_dir>/_merged_hf_cache/global_step_XXX
  2) Replace <exp_dir>/global_step_XXX with merged HF dir (top-level HF files)
  3) Remove cache dir for that step (and cache root if empty)

Notes:
  - Prefers <global_step_XXX>/actor as local_dir when present.
  - Requires: <local_dir>/huggingface/config.json

Examples:
  # Convert all steps under an experiment directory:
  ./convert_verl_all_steps_inplace.sh -i fsdp_gkd_grpo/checkpoints/.../grpo_qwen2p5_0p5b_rollout_8_fp32

  # First run safely (keep backups):
  ./convert_verl_all_steps_inplace.sh -i /path/to/exp --backup

  # Only convert some steps:
  ./convert_verl_all_steps_inplace.sh -i /path/to/exp --steps 300,400,550

  # Keep going even if one step fails:
  ./convert_verl_all_steps_inplace.sh -i /path/to/exp --continue-on-error
EOF
}

PYTHON_EXEC="python"
EXP_DIR=""
OVERWRITE=0
BACKUP=0
CONTINUE_ON_ERROR=0
DRY_RUN=0
STEPS_CSV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input) EXP_DIR="$2"; shift 2 ;;
    --python) PYTHON_EXEC="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift 1 ;;
    --backup) BACKUP=1; shift 1 ;;
    --continue-on-error) CONTINUE_ON_ERROR=1; shift 1 ;;
    --dry-run) DRY_RUN=1; shift 1 ;;
    --steps) STEPS_CSV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$EXP_DIR" ]]; then
  echo "Error: --input <exp_dir> is required" >&2
  usage
  exit 2
fi

if [[ ! -d "$EXP_DIR" ]]; then
  echo "Error: exp_dir not found: $EXP_DIR" >&2
  exit 1
fi

EXP_DIR="$(cd "$EXP_DIR" && pwd)"
CACHE_ROOT="${EXP_DIR}/_merged_hf_cache"
mkdir -p "$CACHE_ROOT"

# Optional filter steps
declare -A STEP_ALLOW=()
if [[ -n "$STEPS_CSV" ]]; then
  IFS=',' read -r -a arr <<<"$STEPS_CSV"
  for s in "${arr[@]}"; do
    s="$(echo "$s" | tr -d ' ')"
    [[ -n "$s" ]] && STEP_ALLOW["$s"]=1
  done
fi

# Collect global_step_* dirs (sorted)
mapfile -t STEP_DIRS < <(find "$EXP_DIR" -maxdepth 1 -type d -name 'global_step_*' | sort)

if [[ "${#STEP_DIRS[@]}" -eq 0 ]]; then
  echo "Error: no global_step_* dirs found under: $EXP_DIR" >&2
  exit 1
fi

extract_step_num() {
  local name
  name="$(basename "$1")"
  echo "${name#global_step_}"
}

merge_one() {
  local step_dir="$1"
  local step_name
  step_name="$(basename "$step_dir")"

  local step_num
  step_num="$(extract_step_num "$step_dir")"
  if [[ -n "$STEPS_CSV" ]] && [[ -z "${STEP_ALLOW[$step_num]+x}" ]]; then
    echo "[skip] ${step_name} (not in --steps)"
    return 0
  fi

  local local_dir="$step_dir"
  if [[ -d "${step_dir}/actor" ]]; then
    local_dir="${step_dir}/actor"
  fi

  local cfg="${local_dir}/huggingface/config.json"
  if [[ ! -f "$cfg" ]]; then
    echo "ERROR: missing HF config for merger: $cfg" >&2
    return 1
  fi

  local out_dir="${CACHE_ROOT}/${step_name}"

  if [[ -d "$out_dir" ]] && [[ "$(ls -A "$out_dir" 2>/dev/null | wc -l | tr -d ' ')" != "0" ]]; then
    if [[ "$OVERWRITE" -eq 1 ]]; then
      echo "[overwrite] rm -rf $out_dir"
      [[ "$DRY_RUN" -eq 1 ]] || rm -rf "$out_dir"
    else
      echo "[cache exists] ${step_name} (skip merge; use --overwrite to regenerate)"
      return 0
    fi
  fi

  echo "[merge] ${step_name}"
  echo "  local_dir:  $local_dir"
  echo "  target_dir: $out_dir"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  "${PYTHON_EXEC}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$local_dir" \
    --target_dir "$out_dir"

  [[ -f "${out_dir}/config.json" ]] || { echo "ERROR: merge output missing config.json: $out_dir" >&2; return 1; }
}

install_one() {
  local step_dir="$1"
  local step_name
  step_name="$(basename "$step_dir")"
  local out_dir="${CACHE_ROOT}/${step_name}"

  local step_num
  step_num="$(extract_step_num "$step_dir")"
  if [[ -n "$STEPS_CSV" ]] && [[ -z "${STEP_ALLOW[$step_num]+x}" ]]; then
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] would replace $step_dir with $out_dir"
    return 0
  fi

  [[ -d "$out_dir" ]] || { echo "ERROR: merged dir missing: $out_dir" >&2; return 1; }

  local parent_dir
  parent_dir="$(dirname "$step_dir")"

  if [[ "$BACKUP" -eq 1 ]]; then
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local backup_dir="${parent_dir}/${step_name}.__bak_${ts}"
    echo "[backup] mv $step_dir -> $backup_dir"
    mv "$step_dir" "$backup_dir"
  else
    echo "[delete] rm -rf $step_dir"
    rm -rf "$step_dir"
  fi

  echo "[install] mv $out_dir -> $step_dir"
  mv "$out_dir" "$step_dir"

  echo "[ok] ${step_name} installed HF checkpoint at: $step_dir"
}

fail_count=0

for d in "${STEP_DIRS[@]}"; do
  echo
  echo "=== $(basename "$d") ==="
  if merge_one "$d"; then
    if ! install_one "$d"; then
      echo "[fail] install failed for $d" >&2
      ((fail_count+=1))
      [[ "$CONTINUE_ON_ERROR" -eq 1 ]] || exit 1
    fi
  else
    echo "[fail] merge failed for $d" >&2
    ((fail_count+=1))
    [[ "$CONTINUE_ON_ERROR" -eq 1 ]] || exit 1
  fi
done

# Remove cache root if empty
if [[ -d "$CACHE_ROOT" ]] && [[ "$(find "$CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')" == "0" ]]; then
  rmdir "$CACHE_ROOT" 2>/dev/null || true
fi

echo
if [[ "$fail_count" -eq 0 ]]; then
  echo "Done. All requested steps converted in place under: $EXP_DIR"
else
  echo "Done with failures: $fail_count (see logs above)."
fi