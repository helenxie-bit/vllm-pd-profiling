#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# KV transfer profiling sweeps for Mooncake 1P1D (or xPyD) disaggregated setup.
#
# Runs three workload sweeps to compare KV transfer metrics (rdma_ms,
# t_transfer_ms, kv_xfer_bandwidth_gbps) under different input lengths,
# request rates, and burstiness values.
#
# Prerequisites: P/D/proxy already running with VLLM_MOONCAKE_PD_PROFILE=1
#
# Usage:
#   export HF_TOKEN=hf_...
#   export VLLM_MOONCAKE_PD_PROFILE=1
#   export MOONCAKE_DEVICE_NAME=mlx5_0
#   # Start P/D/proxy manually (see run_mooncake_connector.sh)
#   bash run_ietf_cat2_workloads.sh
#
# Configuration (env vars):
#   PROFILE_MODE     quick | standard | full  (default: quick)
#   NUM_PROMPTS      Override prompts per run (default: derived from PROFILE_MODE)
#   RUN_SWEEPS       all | input_len | rate | burstiness | comma-separated
#   INPUT_LENS       Comma-separated override for sweep A
#   REQUEST_RATES    Comma-separated override for sweep B
#   BURSTINESS_VALUES Comma-separated override for sweep C
#   RATE_SWEEP_INPUT_LEN   Fixed input len for rate sweep (default: 2048)
#   BURSTINESS_SWEEP_RATE  Fixed rate for burstiness sweep (default: 2)
#   BURSTINESS_SWEEP_INPUT_LEN  Fixed input len for burstiness sweep (default: 2048)
#   OUTPUT_LEN       Decode tokens per request (default: 64)
#   RESULTS_DIR      Output directory (default: ./ietf_cat2_results)
#
# Compare results:
#   find ./ietf_cat2_results -name metrics.json | sort
#   # Key metrics per run: rdma_ms, t_transfer_ms, kv_xfer_bandwidth_gbps

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
PROXY_PORT=${PROXY_PORT:-8000}
PROFILE_DIR=${VLLM_MOONCAKE_PD_PROFILE_DIR:-/tmp/vllm_pd_profile_ietf}
RESULTS_DIR=${RESULTS_DIR:-./ietf_cat2_results}
OUTPUT_LEN=${OUTPUT_LEN:-64}
DEFAULT_BURSTINESS=${DEFAULT_BURSTINESS:-1.0}

PROFILE_MODE=${PROFILE_MODE:-quick}
RUN_SWEEPS=${RUN_SWEEPS:-all}

RATE_SWEEP_INPUT_LEN=${RATE_SWEEP_INPUT_LEN:-2048}
BURSTINESS_SWEEP_RATE=${BURSTINESS_SWEEP_RATE:-2}
BURSTINESS_SWEEP_INPUT_LEN=${BURSTINESS_SWEEP_INPUT_LEN:-2048}

# Mode-derived defaults (overridable via env).
case "$PROFILE_MODE" in
    quick)
        : "${NUM_PROMPTS:=8}"
        : "${INPUT_LENS:=512,2048,8192}"
        : "${REQUEST_RATES:=1,5}"
        : "${BURSTINESS_VALUES:=0.5,5.0}"
        ;;
    standard)
        : "${NUM_PROMPTS:=12}"
        : "${INPUT_LENS:=512,2048,8192,16384}"
        : "${REQUEST_RATES:=1,2,5}"
        : "${BURSTINESS_VALUES:=0.5,1.0,5.0}"
        ;;
    full)
        : "${NUM_PROMPTS:=20}"
        : "${INPUT_LENS:=128,512,2048,4096,8192,16384}"
        : "${REQUEST_RATES:=1,2,5,10}"
        : "${BURSTINESS_VALUES:=0.1,0.5,1.0,5.0,100}"
        ;;
    *)
        echo "Unknown PROFILE_MODE=$PROFILE_MODE (expected quick, standard, or full)"
        exit 1
        ;;
esac

IFS=',' read -ra INPUT_LEN_ARRAY <<< "$INPUT_LENS"
IFS=',' read -ra REQUEST_RATE_ARRAY <<< "$REQUEST_RATES"
IFS=',' read -ra BURSTINESS_ARRAY <<< "$BURSTINESS_VALUES"

mkdir -p "$RESULTS_DIR"

should_run_sweep() {
    local sweep=$1
    [[ "$RUN_SWEEPS" == "all" ]] && return 0
    IFS=',' read -ra enabled <<< "$RUN_SWEEPS"
    for s in "${enabled[@]}"; do
        if [[ "$s" == "$sweep" ]]; then
            return 0
        fi
    done
    return 1
}

run_bench() {
    local tag=$1
    local input_len=$2
    local rate=$3
    local num_prompts=$4
    local burstiness=${5:-$DEFAULT_BURSTINESS}

    local run_dir="$RESULTS_DIR/${tag}_in${input_len}_r${rate}_b${burstiness}_n${num_prompts}"
    mkdir -p "$run_dir"

    if [[ "${VLLM_MOONCAKE_PD_PROFILE:-0}" == "1" ]]; then
        rm -f "$PROFILE_DIR"/*.jsonl
    fi

    echo "=== $tag: input_len=$input_len rate=$rate prompts=$num_prompts burstiness=$burstiness ==="
    vllm bench serve --port "$PROXY_PORT" --seed "$(date +%s)" \
        --backend vllm --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$num_prompts" \
        --request-rate "$rate" \
        --burstiness "$burstiness" \
        2>&1 | tee "$run_dir/bench.log"

    if [[ "${VLLM_MOONCAKE_PD_PROFILE:-0}" == "1" && -d "$PROFILE_DIR" ]]; then
        cp "$PROFILE_DIR"/*.jsonl "$run_dir/" 2>/dev/null || true
        python3 analyze_pd_profile.py \
            --dir "$run_dir" \
            --csv "$run_dir/metrics.csv" \
            --json "$run_dir/metrics.json" \
            | tee "$run_dir/analysis.txt"
        echo "  KV metrics -> $run_dir/metrics.json (rdma_ms, t_transfer_ms, kv_xfer_bandwidth_gbps)"
    fi
}

run_sweep_input_len() {
    echo ">>> Sweep A: KV size scaling (input_len, rate=1, burstiness=$DEFAULT_BURSTINESS)"
    for len in "${INPUT_LEN_ARRAY[@]}"; do
        run_bench "kv_input_len" "$len" 1 "$NUM_PROMPTS" "$DEFAULT_BURSTINESS"
    done
}

run_sweep_rate() {
    echo ">>> Sweep B: Rate / contention (input_len=$RATE_SWEEP_INPUT_LEN, burstiness=$DEFAULT_BURSTINESS)"
    for rate in "${REQUEST_RATE_ARRAY[@]}"; do
        run_bench "kv_rate" "$RATE_SWEEP_INPUT_LEN" "$rate" "$NUM_PROMPTS" "$DEFAULT_BURSTINESS"
    done
}

run_sweep_burstiness() {
    echo ">>> Sweep C: Burstiness (input_len=$BURSTINESS_SWEEP_INPUT_LEN, rate=$BURSTINESS_SWEEP_RATE)"
    for burst in "${BURSTINESS_ARRAY[@]}"; do
        run_bench "kv_burstiness" "$BURSTINESS_SWEEP_INPUT_LEN" "$BURSTINESS_SWEEP_RATE" \
            "$NUM_PROMPTS" "$burst"
    done
}

total_runs=0
if should_run_sweep input_len; then
    total_runs=$((total_runs + ${#INPUT_LEN_ARRAY[@]}))
fi
if should_run_sweep rate; then
    total_runs=$((total_runs + ${#REQUEST_RATE_ARRAY[@]}))
fi
if should_run_sweep burstiness; then
    total_runs=$((total_runs + ${#BURSTINESS_ARRAY[@]}))
fi
total_requests=$((total_runs * NUM_PROMPTS))

echo "KV transfer profiling runner"
echo "  MODEL=$MODEL PROXY_PORT=$PROXY_PORT"
echo "  PROFILE_MODE=$PROFILE_MODE NUM_PROMPTS=$NUM_PROMPTS RUN_SWEEPS=$RUN_SWEEPS"
echo "  PROFILE_DIR=$PROFILE_DIR RESULTS_DIR=$RESULTS_DIR"
echo "  Planned: $total_runs runs, ~$total_requests requests"
echo ""

if should_run_sweep input_len; then
    run_sweep_input_len
fi

if should_run_sweep rate; then
    run_sweep_rate
fi

if should_run_sweep burstiness; then
    run_sweep_burstiness
fi

echo ""
echo "Done. Results in $RESULTS_DIR"
echo "Compare KV transfer metrics:"
echo "  find $RESULTS_DIR -name metrics.json | sort"
