#!/bin/bash
# Monitor CPU/GPU memory and GPU utilization across all training pods
# Usage: bash monitor_memory.sh [interval_seconds] [namespace] [pod_prefix]
#   e.g. bash monitor_memory.sh 10 explore-train hrd-vr-general
# Log is saved to experiments/results/memory_monitor_<timestamp>.log

INTERVAL=${1:-10}
NAMESPACE="explore-train"
PREFIX=${2:-"hrd-3"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../results"
mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/memory_monitor_$(date +%Y%m%d_%H%M%S).log"

PODS=($(kubectl get pods -n "$NAMESPACE" --no-headers -o custom-columns=":metadata.name" 2>/dev/null | grep "^${PREFIX}" | sort -V))

if [ ${#PODS[@]} -eq 0 ]; then
    echo "No pods found with prefix '$PREFIX' in namespace '$NAMESPACE'"
    exit 1
fi

echo "Monitoring ${#PODS[@]} pods every ${INTERVAL}s: ${PODS[*]}"
echo "Logging to: $LOGFILE"
echo "Press Ctrl+C to stop"
echo ""

log_line() {
    local colored="$1"
    local plain="$2"
    echo -e "$colored"
    echo "$plain" >> "$LOGFILE"
}

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    HEADER="======== $TIMESTAMP ========"
    COLUMNS=$(printf "%-30s %7s %7s %5s  |  %-20s  |  %s" "POD" "CPU_USE" "CPU_FR" "CPU%" "GPU_MEM (per card)" "GPU_UTIL avg")
    SEP="--------------------------------------------------------------------------------------------------------------"

    log_line "$HEADER" "$HEADER"
    log_line "$COLUMNS" "$COLUMNS"
    log_line "$SEP" "$SEP"

    for POD in "${PODS[@]}"; do
        # CPU memory + GPU memory + GPU utilization in one kubectl exec
        ALL_INFO=$(kubectl exec -n "$NAMESPACE" "$POD" -- bash -c '
            # CPU memory
            read total _ <<< $(grep MemTotal /proc/meminfo | awk "{print \$2}")
            read avail _ <<< $(grep MemAvailable /proc/meminfo | awk "{print \$2}")
            used=$((total - avail))
            total_gb=$((total / 1048576))
            used_gb=$((used / 1048576))
            avail_gb=$((avail / 1048576))
            pct=$((used * 100 / total))
            echo "CPU:${used_gb}:${avail_gb}:${total_gb}:${pct}"

            # GPU memory (used per card in GB)
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | \
                awk "{printf \"%.0f \", \$1/1024}" | xargs echo "GMEM:"

            # GPU utilization (per card %)
            nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
                awk "BEGIN{s=0;n=0} {s+=\$1;n++} END{printf \"GUTIL:%d:%s\n\", (n>0?s/n:0), (n>0?sprintf(\"%d %d %d %d %d %d %d %d\",0,0,0,0,0,0,0,0):\"?\")}"
            # also get per-card util
            nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
                awk "{printf \"%d \", \$1}" | xargs echo "GUTIL_CARDS:"
        ' 2>/dev/null)

        if [ -n "$ALL_INFO" ]; then
            CPU_LINE=$(echo "$ALL_INFO" | grep "^CPU:")
            GMEM_LINE=$(echo "$ALL_INFO" | grep "^GMEM:")
            GUTIL_CARDS_LINE=$(echo "$ALL_INFO" | grep "^GUTIL_CARDS:")

            IFS=':' read -r _ USED FREE TOTAL PCT <<< "$CPU_LINE"
            GPU_MEM=$(echo "$GMEM_LINE" | sed 's/^GMEM: *//' | awk '{for(i=1;i<=NF;i++) printf "%sG ", $i}')
            GPU_UTILS=$(echo "$GUTIL_CARDS_LINE" | sed 's/^GUTIL_CARDS: *//')

            # Calculate average GPU util
            AVG_UTIL=$(echo "$GPU_UTILS" | awk '{s=0;n=0; for(i=1;i<=NF;i++){s+=$i;n++} printf "%d", (n>0?s/n:0)}')

            # Color for CPU
            if [ "$PCT" -gt 80 ]; then
                CPU_COLOR="\033[0;31m"
            elif [ "$PCT" -gt 60 ]; then
                CPU_COLOR="\033[0;33m"
            else
                CPU_COLOR="\033[0;32m"
            fi

            # Color for GPU util
            if [ "$AVG_UTIL" -gt 80 ]; then
                GPU_COLOR="\033[0;32m"
            elif [ "$AVG_UTIL" -gt 20 ]; then
                GPU_COLOR="\033[0;33m"
            else
                GPU_COLOR="\033[0;31m"
            fi

            PLAIN=$(printf "%-30s %5sGB %5sGB %4s%%  |  %-20s  |  avg %3s%% [%s]" \
                "$POD" "$USED" "$FREE" "$PCT" "$GPU_MEM" "$AVG_UTIL" "$GPU_UTILS")
            COLORED=$(printf "%-30s %5sGB %5sGB ${CPU_COLOR}%4s%%\033[0m  |  %-20s  |  ${GPU_COLOR}avg %3s%%\033[0m [%s]" \
                "$POD" "$USED" "$FREE" "$PCT" "$GPU_MEM" "$AVG_UTIL" "$GPU_UTILS")

            log_line "$COLORED" "$PLAIN"
        else
            log_line "$(printf '%-30s %s' "$POD" 'UNREACHABLE')" "$(printf '%-30s %s' "$POD" 'UNREACHABLE')"
        fi
    done

    log_line "" ""
    sleep "$INTERVAL"
done
