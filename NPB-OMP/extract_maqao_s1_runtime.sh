#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default MAQAO directories requested by user.
DEFAULT_DIRS=(
  "maqao-1"
  "maqao_with_compil_option-2"
  "maqao_optimized_with_ompoptionforsleepingthreads-4"
  "maqao_optimized_vectorisation-7"
  "maqao_newflags-8"
)

# Accept custom directories as arguments, else fallback to defaults.
if [ "$#" -gt 0 ]; then
  DIRS=("$@")
else
  DIRS=("${DEFAULT_DIRS[@]}")
fi

printf "%-52s | %-16s | %-16s | %s\n" "Directory" "time (s)" "profiled_time" "CSV path"
printf -- "-----------------------------------------------------------------------------------------------\n"

for d in "${DIRS[@]}"; do
  # Graceful alias for common typo "omptoption".
  if [ "$d" = "maqao_optimized_with_omptoptionforsleepingthreads-4" ]; then
    d="maqao_optimized_with_ompoptionforsleepingthreads-4"
  fi

  base="${ROOT_DIR}/${d}"
  if [ ! -d "$base" ]; then
    printf "%-52s | %-16s | %-16s | %s\n" "$d" "N/A" "N/A" "directory not found"
    continue
  fi

  csv_path="$(find "$base" -path "*/ov_cgC_S1_6T/shared/run_0/expert_run.csv" | head -n 1 || true)"
  if [ -z "$csv_path" ]; then
    printf "%-52s | %-16s | %-16s | %s\n" "$d" "N/A" "N/A" "expert_run.csv not found"
    continue
  fi

  # Parse second line fields: time;profiled_time;...
  row="$(sed -n '2p' "$csv_path" || true)"
  time_s="$(awk -F';' 'NR==2 {print $1}' "$csv_path")"
  profiled_s="$(awk -F';' 'NR==2 {print $2}' "$csv_path")"

  if [ -z "$row" ] || [ -z "$time_s" ] || [ -z "$profiled_s" ]; then
    printf "%-52s | %-16s | %-16s | %s\n" "$d" "N/A" "N/A" "invalid CSV content"
    continue
  fi

  printf "%-52s | %-16s | %-16s | %s\n" "$d" "$time_s" "$profiled_s" "$csv_path"
done
