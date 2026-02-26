#!/usr/bin/env bash
set -euo pipefail

# Always run from this script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CLASS="C"
BIN="./bin/cg.${CLASS}"
XP_PREFIX="ov_cg${CLASS}"

# Build
make clean
make cg CLASS="${CLASS}"
ls -lh "${BIN}"

# Environnement stable (important pour S1 et WS)
export OMP_WAIT_POLICY=active
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE=static


# MAQAO/LPROF requires perf + ptrace permissions
perf_paranoid="$(cat /proc/sys/kernel/perf_event_paranoid)"
ptrace_scope="$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo 0)"
if [ "${perf_paranoid}" -gt 2 ] || [ "${ptrace_scope}" -gt 0 ]; then
  echo "ERREUR: MAQAO ne peut pas profiler sur cette machine avec la conf actuelle."
  echo "kernel.perf_event_paranoid=${perf_paranoid} (requis <= 2)"
  echo "kernel.yama.ptrace_scope=${ptrace_scope} (requis = 0)"
  echo "Commande temporaire (sudo):"
  echo "  sudo sysctl -w kernel.perf_event_paranoid=2 kernel.yama.ptrace_scope=0"
  exit 1
fi

# Mesures rapides (temps + RAM)
#for t in 1 2 3 6; do
#  echo "=== OMP_NUM_THREADS=$t ==="
#  export OMP_NUM_THREADS=$t
#  /usr/bin/time -v "${BIN}"
#done

# MAQAO OneView - R1 (normal) 1 thread
maqao oneview --replace -R1 --create-report=one \
  --executable="${BIN}" --run-command="<executable>" \
  --lprof-params="-exclude-kernel" \
  --envv_OMP_NUM_THREADS=1 \
  -xp="${XP_PREFIX}_R1_1T"

# MAQAO OneView - R1 (normal) 6 threads
maqao oneview --replace -R1 --create-report=one \
  --executable="${BIN}" --run-command="<executable>" \
  --lprof-params="-exclude-kernel" \
  --envv_OMP_NUM_THREADS=6 \
  -xp="${XP_PREFIX}_R1_6T"

# MAQAO OneView - S1 (stabilité) 6 threads, 15 runs
maqao oneview --replace -S1 --create-report=one \
  --executable="${BIN}" --run-command="<executable>" \
  --lprof-params="-exclude-kernel" \
  --envv_OMP_NUM_THREADS=6 \
  -rep=15 \
  -xp="${XP_PREFIX}_S1_6T"

# MAQAO OneView - WS (strong scaling): base=1 thread + multiruns 2/3/6
#maqao oneview --replace -R1 -WS=strong --create-report=one \
#  --executable="${BIN}" --run-command="<executable>" \
#  --lprof-params="-exclude-kernel" \
#  --envv_OMP_NUM_THREADS=1 \
#  --multiruns-params='{{name="2T", environment_variables={{name="OMP_NUM_THREADS", value="2"}}}, {name="3T", environment_variables={{name="OMP_NUM_THREADS", value="3"}}}, {name="6T", environment_variables={{name="OMP_NUM_THREADS", value="6"}}}}' \
#  -xp="${XP_PREFIX}_WS"