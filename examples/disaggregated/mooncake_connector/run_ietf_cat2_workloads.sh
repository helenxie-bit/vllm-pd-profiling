#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Run IETF draft-calabria-bmwg-ai-fabric-inference-bench-04
# Test Category 2 workloads against Mooncake 1P1D (or xPyD) setup.
#
# Prerequisites: P/D/proxy already running with VLLM_MOONCAKE_PD_PROFILE=1
#
# Usage:
#   export HF_TOKEN=hf_...
#   export MOONCAKE_DEVICE_NAME=mlx5_0
#   bash run_mooncake_connector.sh &   # or start P/D/proxy manually
#   bash run_ietf_cat2_workloads.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
PROXY_PORT=${PROXY_PORT:-8000}
PROFILE_DIR=${VLLM_MOONCAKE_PD_PROFILE_DIR:-/tmp/vllm_pd_profile_ietf}
RESULTS_DIR=${RESULTS_DIR:-./ietf_cat2_results}
NUM_TRIALS=${NUM_TRIALS:-20}
OUTPUT_LEN=${OUTPUT_LEN:-64}

mkdir -p "$RESULTS_DIR"

run_bench() {
    local tag=$1
    local input_len=$2
    local rate=$3
    local num_prompts=$4
    local burstiness=${5:-100}

    local run_dir="$RESULTS_DIR/${tag}_in${input_len}_r${rate}"
    mkdir -p "$run_dir"

    if [[ "${VLLM_MOONCAKE_PD_PROFILE:-0}" == "1" ]]; then
        rm -f "$PROFILE_DIR"/*.jsonl
    fi

    echo "=== $tag: input_len=$input_len rate=$rate prompts=$num_prompts ==="
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
    fi
}

echo "IETF Cat-2 workload runner"
echo "  MODEL=$MODEL PROXY_PORT=$PROXY_PORT"
echo "  PROFILE_DIR=$PROFILE_DIR RESULTS_DIR=$RESULTS_DIR"
echo ""

# ---------------------------------------------------------------------------
# Section 6.1: TTFT vs prompt length (128..16384 tokens)
# Minimum 20 trials per prompt length per the draft.
# ---------------------------------------------------------------------------
echo ">>> Section 6.1: End-to-End Disaggregated TTFT (prompt length sweep)"
for LEN in 128 512 1024 2048 4096 8192 16384; do
    run_bench "cat6.1" "$LEN" 1 "$NUM_TRIALS"
done

# ---------------------------------------------------------------------------
# Section 6.4: Prefill queue depth / oversubscription
# Increase request rate to oversubscribe prefill (1.0x → 2.0x).
# Adjust PREFILL_OVERSUB_BASE if you know your 1P saturation rate.
# ---------------------------------------------------------------------------
echo ">>> Section 6.4: Prefill queue depth impact"
PREFILL_OVERSUB_BASE=${PREFILL_OVERSUB_BASE:-2}
for MULT in 1.0 1.25 1.5 1.75 2.0; do
    RATE=$(python3 -c "print(${PREFILL_OVERSUB_BASE} * ${MULT})")
    run_bench "cat6.4" 4096 "$RATE" "$NUM_TRIALS"
done

# ---------------------------------------------------------------------------
# Section 6.2 partial: sustained load at fixed 1P1D (xPyD ratio sweep
# requires multi-GPU topology changes — run manually for 2P10D etc.)
# ---------------------------------------------------------------------------
echo ">>> Section 6.2 partial: sustained request stream (1P1D baseline)"
for RATE in 1 2 5 10; do
    run_bench "cat6.2" 2048 "$RATE" 100
done

echo ""
echo "Done. Results in $RESULTS_DIR"
echo "Aggregate: find $RESULTS_DIR -name metrics.json"
