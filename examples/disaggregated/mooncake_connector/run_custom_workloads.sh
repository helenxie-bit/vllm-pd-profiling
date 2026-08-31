#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# KV transfer profiling sweeps for Mooncake 1P1D (or xPyD) disaggregated setup.
#
# UPDATED VERSION
# Every substantive modification from the original script is marked with:
#   # NEW:
#   # CHANGED:
#
# Experimental philosophy:
#   quick    = smoke test / correctness check
#   standard = clean one-factor-at-a-time characterization with repetitions
#   full     = broader stress/tail characterization with more samples
#
# Sweeps input length, request rate, burstiness, output length, max concurrency,
# and ISL/OSL heterogeneity to compare KV transfer metrics
# (rdma_ms, t_transfer_ms, kv_xfer_bandwidth_gbps).
#
# Prerequisites: P/D/proxy already running with VLLM_MOONCAKE_PD_PROFILE=1
#
# Usage:
#   export HF_TOKEN=hf_...
#   export VLLM_MOONCAKE_PD_PROFILE=1
#   export MOONCAKE_DEVICE_NAME=mlx5_0
#   # Start P/D/proxy manually (see run_mooncake_connector.sh)
#   bash run_custom_workloads_updated.sh
#
# Configuration (env vars):
#   PROFILE_MODE       quick | standard | full  (default: quick)
#   NUM_PROMPTS        Override measured prompts per run
#   REPEATS            Override repetitions per parameter point
#   RUN_SWEEPS         all | input_len | rate | burstiness | output_len |
#                      concurrency | heterogeneity | comma-separated
#   INPUT_LENS / REQUEST_RATES / BURSTINESS_VALUES
#   OUTPUT_LENS / MAX_CONCURRENCIES
#   HETEROGENEITY_CASES  Comma-separated input_rr:output_rr pairs, e.g.
#                        0.0:0.0,0.5:0.0,0.0:0.5,0.5:0.5
#   DEFAULT_OUTPUT_LEN
#   DEFAULT_INPUT_RANGE_RATIO / DEFAULT_OUTPUT_RANGE_RATIO
#   RATE_SWEEP_* / BURSTINESS_SWEEP_* / OUTPUT_LEN_SWEEP_* /
#   CONCURRENCY_SWEEP_* / HETEROGENEITY_SWEEP_*
#   SEED               Base RNG seed; repetition r uses SEED + r - 1
#   NUM_WARMS          Separate warmup requests before each measured run
#   SEPARATE_WARMUP    1 = warm up separately, then truncate profiling marks
#                      before measured requests (default: 1)
#   SAVE_VLLM_DETAILS  1 = save vLLM per-request benchmark JSON (default: 1)
#   PLOT_TIMELINE      1 = save vLLM request timeline HTML (default: 0)
#   PLOT_DATASET_STATS 1 = save dataset token-length plot (default: 0)
#   RESULTS_DIR        Output directory (default: ./custom_workload_results)
#
# Compare results:
#   find ./custom_workload_results -name metrics.json | sort
#   # Key metrics per run: rdma_ms, t_transfer_ms, kv_xfer_bandwidth_gbps

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
PROXY_PORT=${PROXY_PORT:-8000}
PROFILE_DIR=${VLLM_MOONCAKE_PD_PROFILE_DIR:-/tmp/vllm_pd_profile_custom}
RESULTS_DIR=${RESULTS_DIR:-./custom_workload_results}
SEED=${SEED:-42}
NUM_WARMS=${NUM_WARMS:-3}
DEFAULT_BURSTINESS=${DEFAULT_BURSTINESS:-1.0}

# CHANGED: split the old scalar DEFAULT_RANGE_RATIO into independent ISL/OSL
# controls so input-size heterogeneity and decode-length heterogeneity can be
# isolated experimentally.
DEFAULT_INPUT_RANGE_RATIO=${DEFAULT_INPUT_RANGE_RATIO:-0.0}
DEFAULT_OUTPUT_RANGE_RATIO=${DEFAULT_OUTPUT_RANGE_RATIO:-0.0}

# Fixed output len when not sweeping output_len (OUTPUT_LEN alias kept).
if [[ -n "${OUTPUT_LEN:-}" ]]; then
    DEFAULT_OUTPUT_LEN=${DEFAULT_OUTPUT_LEN:-$OUTPUT_LEN}
fi
DEFAULT_OUTPUT_LEN=${DEFAULT_OUTPUT_LEN:-64}

PROFILE_MODE=${PROFILE_MODE:-quick}
RUN_SWEEPS=${RUN_SWEEPS:-all}

# NEW: repetitions and clean warmup behavior.
SEPARATE_WARMUP=${SEPARATE_WARMUP:-1}
SAVE_VLLM_DETAILS=${SAVE_VLLM_DETAILS:-1}
PLOT_TIMELINE=${PLOT_TIMELINE:-0}
PLOT_DATASET_STATS=${PLOT_DATASET_STATS:-0}

# CHANGED: contention-oriented sweeps use a more mixed P:D workload by default
# (output=256), while the clean KV-size sweep keeps DEFAULT_OUTPUT_LEN=64.
RATE_SWEEP_INPUT_LEN=${RATE_SWEEP_INPUT_LEN:-2048}
RATE_SWEEP_OUTPUT_LEN=${RATE_SWEEP_OUTPUT_LEN:-256}
BURSTINESS_SWEEP_RATE=${BURSTINESS_SWEEP_RATE:-2}
BURSTINESS_SWEEP_INPUT_LEN=${BURSTINESS_SWEEP_INPUT_LEN:-2048}
BURSTINESS_SWEEP_OUTPUT_LEN=${BURSTINESS_SWEEP_OUTPUT_LEN:-256}
OUTPUT_LEN_SWEEP_INPUT_LEN=${OUTPUT_LEN_SWEEP_INPUT_LEN:-2048}
OUTPUT_LEN_SWEEP_RATE=${OUTPUT_LEN_SWEEP_RATE:-1}
CONCURRENCY_SWEEP_INPUT_LEN=${CONCURRENCY_SWEEP_INPUT_LEN:-2048}
# CHANGED: request_rate=inf + max_concurrency isolates offered concurrency more
# cleanly than finite request-rate + max-concurrency, which can throttle at the client.
CONCURRENCY_SWEEP_RATE=${CONCURRENCY_SWEEP_RATE:-inf}
CONCURRENCY_SWEEP_OUTPUT_LEN=${CONCURRENCY_SWEEP_OUTPUT_LEN:-256}
CONCURRENCY_SWEEP_BURSTINESS=${CONCURRENCY_SWEEP_BURSTINESS:-1.0}
HETEROGENEITY_SWEEP_INPUT_LEN=${HETEROGENEITY_SWEEP_INPUT_LEN:-2048}
HETEROGENEITY_SWEEP_OUTPUT_LEN=${HETEROGENEITY_SWEEP_OUTPUT_LEN:-256}
HETEROGENEITY_SWEEP_RATE=${HETEROGENEITY_SWEEP_RATE:-2}

# CHANGED: modes now represent experimental purpose, not just "more points".
# All values remain overridable through environment variables.
case "$PROFILE_MODE" in
    quick)
        # Smoke test: small sample count, one repetition, baseline + extremes.
        : "${NUM_PROMPTS:=30}"
        : "${REPEATS:=1}"
        : "${INPUT_LENS:=512,4096,16384}"
        : "${REQUEST_RATES:=1,5}"
        # CHANGED: include Poisson baseline (1.0); lower value is more bursty.
        : "${BURSTINESS_VALUES:=0.3,1.0}"
        : "${OUTPUT_LENS:=32,256}"
        : "${MAX_CONCURRENCIES:=1,8,32}"
        # NEW: input-only, output-only, and joint heterogeneity are explicit.
        : "${HETEROGENEITY_CASES:=0.0:0.0,0.5:0.0,0.0:0.5,0.5:0.5}"
        ;;
    standard)
        # Main characterization: moderate sample count + 3 repetitions.
        : "${NUM_PROMPTS:=100}"
        : "${REPEATS:=3}"
        : "${INPUT_LENS:=256,1024,4096,16384}"
        : "${REQUEST_RATES:=1,2,5}"
        : "${BURSTINESS_VALUES:=0.3,1.0,3.0}"
        : "${OUTPUT_LENS:=32,128,512,1024}"
        : "${MAX_CONCURRENCIES:=1,4,16,64}"
        : "${HETEROGENEITY_CASES:=0.0:0.0,0.5:0.0,0.0:0.5,0.5:0.5}"
        ;;
    full)
        # Stress/tail characterization: more samples and repetitions.
        : "${NUM_PROMPTS:=300}"
        : "${REPEATS:=3}"
        : "${INPUT_LENS:=128,512,2048,8192,16384}"
        : "${REQUEST_RATES:=1,2,5,10}"
        # CHANGED: remove 100 (near-deterministic extreme) from the normal suite.
        : "${BURSTINESS_VALUES:=0.1,0.3,1.0,3.0,5.0}"
        : "${OUTPUT_LENS:=16,64,256,1024}"
        : "${MAX_CONCURRENCIES:=1,4,16,64}"
        # NEW: stronger heterogeneity case added for stress testing.
        : "${HETEROGENEITY_CASES:=0.0:0.0,0.5:0.0,0.0:0.5,0.5:0.5,0.8:0.8}"
        ;;
    *)
        echo "Unknown PROFILE_MODE=$PROFILE_MODE (expected quick, standard, or full)"
        exit 1
        ;;
esac

IFS=',' read -ra INPUT_LEN_ARRAY <<< "$INPUT_LENS"
IFS=',' read -ra REQUEST_RATE_ARRAY <<< "$REQUEST_RATES"
IFS=',' read -ra BURSTINESS_ARRAY <<< "$BURSTINESS_VALUES"
IFS=',' read -ra OUTPUT_LEN_ARRAY <<< "$OUTPUT_LENS"
IFS=',' read -ra MAX_CONCURRENCY_ARRAY <<< "$MAX_CONCURRENCIES"
# CHANGED: replaces the old scalar RANGE_RATIO_ARRAY.
IFS=',' read -ra HETEROGENEITY_ARRAY <<< "$HETEROGENEITY_CASES"

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

truncate_profile_marks() {
    if [[ "${VLLM_MOONCAKE_PD_PROFILE:-0}" != "1" ]]; then
        return 0
    fi
    if [[ ! -d "$PROFILE_DIR" ]]; then
        echo "ERROR: PROFILE_DIR=$PROFILE_DIR does not exist." \
            "Start P/D/proxy with VLLM_MOONCAKE_PD_PROFILE=1 first." >&2
        exit 1
    fi
    # Truncate in place (do not rm): P/D/proxy keep append FDs open; unlinking
    # would send later marks to deleted inodes and leave PROFILE_DIR empty.
    shopt -s nullglob
    for f in "$PROFILE_DIR"/*.jsonl; do
        : > "$f"
    done
    shopt -u nullglob
}

# NEW: construct vLLM's JSON form for --random-range-ratio so ISL and OSL
# variability can be controlled independently.
range_ratio_json() {
    local input_rr=$1
    local output_rr=$2
    printf '{"input":%s,"output":%s}' "$input_rr" "$output_rr"
}

# NEW: compact filesystem-safe label for a heterogeneity setting.
range_ratio_label() {
    local input_rr=$1
    local output_rr=$2
    printf 'irr%s_orr%s' "$input_rr" "$output_rr"
}

# NEW: warm up in a separate benchmark invocation, then truncate profiling marks.
# This prevents warmup requests from contaminating measured PD profile JSONL.
run_separate_warmup() {
    local input_len=$1
    local output_len=$2
    local input_rr=$3
    local output_rr=$4
    local run_seed=$5
    local log_file=$6

    if [[ "$NUM_WARMS" -le 0 ]]; then
        return 0
    fi

    local rr_json
    rr_json=$(range_ratio_json "$input_rr" "$output_rr")

    local -a warm_cmd=(
        vllm bench serve
        --port "$PROXY_PORT"
        --seed "$run_seed"
        --backend vllm
        --model "$MODEL"
        --dataset-name random
        --random-input-len "$input_len"
        --random-output-len "$output_len"
        --random-range-ratio "$rr_json"
        --num-prompts "$NUM_WARMS"
        --request-rate inf
        --max-concurrency 1
        --ignore-eos
        --num-warmups 0
    )

    echo "  Warmup: $NUM_WARMS requests (separate, concurrency=1; profile marks discarded)"
    "${warm_cmd[@]}" >"$log_file" 2>&1

    # Critical ordering: discard all warmup marks immediately before measurement.
    truncate_profile_marks
}

# CHANGED signature:
# run_bench TAG INPUT_LEN RATE NUM_PROMPTS BURSTINESS OUTPUT_LEN INPUT_RR OUTPUT_RR [MAX_CONCURRENCY]
run_bench() {
    local tag=$1
    local input_len=$2
    local rate=$3
    local num_prompts=$4
    local burstiness=${5:-$DEFAULT_BURSTINESS}
    local output_len=${6:-$DEFAULT_OUTPUT_LEN}
    local input_rr=${7:-$DEFAULT_INPUT_RANGE_RATIO}
    local output_rr=${8:-$DEFAULT_OUTPUT_RANGE_RATIO}
    local max_concurrency=${9:-}

    local rr_json rr_label
    rr_json=$(range_ratio_json "$input_rr" "$output_rr")
    rr_label=$(range_ratio_label "$input_rr" "$output_rr")

    # NEW: repeat every parameter point. Seeds vary across repetitions.
    local rep
    for ((rep=1; rep<=REPEATS; rep++)); do
        local run_seed=$((SEED + rep - 1))
        local run_dir="$RESULTS_DIR/${tag}_in${input_len}_out${output_len}_r${rate}_b${burstiness}_${rr_label}_n${num_prompts}_rep${rep}_seed${run_seed}"
        if [[ -n "$max_concurrency" ]]; then
            run_dir="${run_dir}_c${max_concurrency}"
        fi
        mkdir -p "$run_dir"

        # Start each repetition from clean profiling files.
        truncate_profile_marks

        # NEW: optionally perform warmup separately, then clear only its marks.
        if [[ "$SEPARATE_WARMUP" == "1" ]]; then
            run_separate_warmup "$input_len" "$output_len" "$input_rr" "$output_rr" \
                "$run_seed" "$run_dir/warmup.log"
        fi

        local -a cmd=(
            vllm bench serve
            --port "$PROXY_PORT"
            --seed "$run_seed"
            --backend vllm
            --model "$MODEL"
            --dataset-name random
            --random-input-len "$input_len"
            --random-output-len "$output_len"
            --random-range-ratio "$rr_json"
            --num-prompts "$num_prompts"
            --request-rate "$rate"
            --burstiness "$burstiness"
            --ignore-eos
        )

        # CHANGED: if warmup is separate, measured invocation contains zero warmups.
        if [[ "$SEPARATE_WARMUP" == "1" ]]; then
            cmd+=(--num-warmups 0)
        else
            cmd+=(--num-warmups "$NUM_WARMS")
        fi

        if [[ -n "$max_concurrency" ]]; then
            cmd+=(--max-concurrency "$max_concurrency")
        fi

        # NEW: save vLLM request-level metrics alongside custom PD metrics.
        if [[ "$SAVE_VLLM_DETAILS" == "1" ]]; then
            cmd+=(
                --save-result
                --save-detailed
                --result-dir "$run_dir"
                --result-filename "vllm_bench.json"
                --percentile-metrics ttft,tpot,itl,e2el
                --metric-percentiles 50,90,95,99
            )
        fi
        if [[ "$PLOT_TIMELINE" == "1" ]]; then
            cmd+=(--plot-timeline)
        fi
        if [[ "$PLOT_DATASET_STATS" == "1" ]]; then
            cmd+=(--plot-dataset-stats)
        fi

        echo "=== $tag rep=$rep/$REPEATS: in=$input_len out=$output_len rate=$rate" \
            "burst=$burstiness input_rr=$input_rr output_rr=$output_rr" \
            "prompts=$num_prompts concurrency=${max_concurrency:-none}" \
            "seed=$run_seed separate_warmup=$SEPARATE_WARMUP ignore_eos=1 ==="
        "${cmd[@]}" 2>&1 | tee "$run_dir/bench.log"

        if [[ "${VLLM_MOONCAKE_PD_PROFILE:-0}" == "1" ]]; then
            shopt -s nullglob
            local jsonl_files=("$PROFILE_DIR"/*.jsonl)
            shopt -u nullglob
            if [[ ${#jsonl_files[@]} -eq 0 ]]; then
                echo "ERROR: no *.jsonl under PROFILE_DIR=$PROFILE_DIR after run $tag." \
                    "Ensure P/D/proxy were started with VLLM_MOONCAKE_PD_PROFILE=1" \
                    "and VLLM_MOONCAKE_PD_PROFILE_DIR=$PROFILE_DIR." >&2
                exit 1
            fi
            local nonempty=0
            local f
            for f in "${jsonl_files[@]}"; do
                if [[ -s "$f" ]]; then
                    nonempty=1
                    break
                fi
            done
            if [[ "$nonempty" -eq 0 ]]; then
                echo "ERROR: *.jsonl under PROFILE_DIR=$PROFILE_DIR are empty after run $tag." \
                    "Marks were not written (servers may still be writing to unlinked" \
                    "files from a prior rm; restart P/D/proxy and retry)." >&2
                exit 1
            fi
            cp "${jsonl_files[@]}" "$run_dir/"
            python3 analyze_pd_profile.py \
                --dir "$run_dir" \
                --csv "$run_dir/metrics.csv" \
                --json "$run_dir/metrics.json" \
                | tee "$run_dir/analysis.txt"
            if [[ ! -f "$run_dir/metrics.json" ]]; then
                echo "ERROR: analyze_pd_profile.py did not write $run_dir/metrics.json" \
                    "(no transferable marks in copied JSONL)." >&2
                exit 1
            fi
            echo "  KV metrics -> $run_dir/metrics.json (rdma_ms, t_transfer_ms, kv_xfer_bandwidth_gbps)"
            if [[ "$SEPARATE_WARMUP" == "1" ]]; then
                echo "  Warmup marks excluded: warmup ran separately and profile files were truncated before measurement."
            else
                echo "  WARNING: profile marks include --num-warmups=$NUM_WARMS plus measured prompts."
            fi
        fi
    done
}

run_sweep_input_len() {
    # CHANGED: force max_concurrency=1 to isolate intrinsic input/KV-size scaling.
    echo ">>> Sweep A: CLEAN KV size scaling (input_len; concurrency=1; out=$DEFAULT_OUTPUT_LEN)"
    for len in "${INPUT_LEN_ARRAY[@]}"; do
        run_bench "kv_input_len" "$len" 1 "$NUM_PROMPTS" "$DEFAULT_BURSTINESS" \
            "$DEFAULT_OUTPUT_LEN" "$DEFAULT_INPUT_RANGE_RATIO" \
            "$DEFAULT_OUTPUT_RANGE_RATIO" 1
    done
}

run_sweep_rate() {
    echo ">>> Sweep B: Rate / contention (input=$RATE_SWEEP_INPUT_LEN, out=$RATE_SWEEP_OUTPUT_LEN)"
    for rate in "${REQUEST_RATE_ARRAY[@]}"; do
        run_bench "kv_rate" "$RATE_SWEEP_INPUT_LEN" "$rate" "$NUM_PROMPTS" \
            "$DEFAULT_BURSTINESS" "$RATE_SWEEP_OUTPUT_LEN" \
            "$DEFAULT_INPUT_RANGE_RATIO" "$DEFAULT_OUTPUT_RANGE_RATIO"
    done
}

run_sweep_burstiness() {
    echo ">>> Sweep C: Burstiness (input=$BURSTINESS_SWEEP_INPUT_LEN, out=$BURSTINESS_SWEEP_OUTPUT_LEN, rate=$BURSTINESS_SWEEP_RATE)"
    for burst in "${BURSTINESS_ARRAY[@]}"; do
        run_bench "kv_burstiness" "$BURSTINESS_SWEEP_INPUT_LEN" "$BURSTINESS_SWEEP_RATE" \
            "$NUM_PROMPTS" "$burst" "$BURSTINESS_SWEEP_OUTPUT_LEN" \
            "$DEFAULT_INPUT_RANGE_RATIO" "$DEFAULT_OUTPUT_RANGE_RATIO"
    done
}

run_sweep_output_len() {
    # CHANGED: force max_concurrency=1 to make this a clean output/decode scaling baseline.
    echo ">>> Sweep D: CLEAN output/decode scaling (input=$OUTPUT_LEN_SWEEP_INPUT_LEN; concurrency=1)"
    for out in "${OUTPUT_LEN_ARRAY[@]}"; do
        run_bench "kv_output_len" "$OUTPUT_LEN_SWEEP_INPUT_LEN" "$OUTPUT_LEN_SWEEP_RATE" \
            "$NUM_PROMPTS" "$DEFAULT_BURSTINESS" "$out" \
            "$DEFAULT_INPUT_RANGE_RATIO" "$DEFAULT_OUTPUT_RANGE_RATIO" 1
    done
}

run_sweep_concurrency() {
    # CHANGED: default request-rate is inf so max_concurrency is the primary offered-load control.
    echo ">>> Sweep E: Offered concurrency (input=$CONCURRENCY_SWEEP_INPUT_LEN, out=$CONCURRENCY_SWEEP_OUTPUT_LEN, rate=$CONCURRENCY_SWEEP_RATE)"
    for c in "${MAX_CONCURRENCY_ARRAY[@]}"; do
        run_bench "kv_concurrency" "$CONCURRENCY_SWEEP_INPUT_LEN" "$CONCURRENCY_SWEEP_RATE" \
            "$NUM_PROMPTS" "$CONCURRENCY_SWEEP_BURSTINESS" "$CONCURRENCY_SWEEP_OUTPUT_LEN" \
            "$DEFAULT_INPUT_RANGE_RATIO" "$DEFAULT_OUTPUT_RANGE_RATIO" "$c"
    done
}

# CHANGED: replaces scalar range_ratio sweep with explicit input-only,
# output-only, and joint heterogeneity cases.
run_sweep_heterogeneity() {
    echo ">>> Sweep F: Length heterogeneity (input=$HETEROGENEITY_SWEEP_INPUT_LEN, out=$HETEROGENEITY_SWEEP_OUTPUT_LEN, rate=$HETEROGENEITY_SWEEP_RATE)"
    local case_spec input_rr output_rr
    for case_spec in "${HETEROGENEITY_ARRAY[@]}"; do
        IFS=':' read -r input_rr output_rr <<< "$case_spec"
        if [[ -z "${input_rr:-}" || -z "${output_rr:-}" ]]; then
            echo "ERROR: invalid HETEROGENEITY_CASES entry '$case_spec'; expected input_rr:output_rr" >&2
            exit 1
        fi
        run_bench "kv_heterogeneity" "$HETEROGENEITY_SWEEP_INPUT_LEN" \
            "$HETEROGENEITY_SWEEP_RATE" "$NUM_PROMPTS" "$DEFAULT_BURSTINESS" \
            "$HETEROGENEITY_SWEEP_OUTPUT_LEN" "$input_rr" "$output_rr"
    done
}

# NEW: total count includes repetitions; warmups are separate when enabled.
total_points=0
if should_run_sweep input_len; then
    total_points=$((total_points + ${#INPUT_LEN_ARRAY[@]}))
fi
if should_run_sweep rate; then
    total_points=$((total_points + ${#REQUEST_RATE_ARRAY[@]}))
fi
if should_run_sweep burstiness; then
    total_points=$((total_points + ${#BURSTINESS_ARRAY[@]}))
fi
if should_run_sweep output_len; then
    total_points=$((total_points + ${#OUTPUT_LEN_ARRAY[@]}))
fi
if should_run_sweep concurrency; then
    total_points=$((total_points + ${#MAX_CONCURRENCY_ARRAY[@]}))
fi
if should_run_sweep heterogeneity; then
    total_points=$((total_points + ${#HETEROGENEITY_ARRAY[@]}))
fi

total_runs=$((total_points * REPEATS))
measured_requests=$((total_runs * NUM_PROMPTS))
if [[ "$SEPARATE_WARMUP" == "1" ]]; then
    warmup_requests=$((total_runs * NUM_WARMS))
else
    warmup_requests=$((total_runs * NUM_WARMS))
fi

echo "KV transfer profiling runner (UPDATED)"
echo "  MODEL=$MODEL PROXY_PORT=$PROXY_PORT"
echo "  PROFILE_MODE=$PROFILE_MODE NUM_PROMPTS=$NUM_PROMPTS REPEATS=$REPEATS RUN_SWEEPS=$RUN_SWEEPS"
echo "  BASE_SEED=$SEED NUM_WARMS=$NUM_WARMS SEPARATE_WARMUP=$SEPARATE_WARMUP"
echo "  DEFAULT_OUTPUT_LEN=$DEFAULT_OUTPUT_LEN"
echo "  DEFAULT_INPUT_RANGE_RATIO=$DEFAULT_INPUT_RANGE_RATIO DEFAULT_OUTPUT_RANGE_RATIO=$DEFAULT_OUTPUT_RANGE_RATIO"
echo "  PROFILE_DIR=$PROFILE_DIR RESULTS_DIR=$RESULTS_DIR"
echo "  Planned: $total_points parameter points x $REPEATS = $total_runs measured runs"
echo "  Requests: ~$measured_requests measured + ~$warmup_requests warmup"
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

if should_run_sweep output_len; then
    run_sweep_output_len
fi

if should_run_sweep concurrency; then
    run_sweep_concurrency
fi

if should_run_sweep heterogeneity; then
    run_sweep_heterogeneity
fi

echo ""
echo "Done. Results in $RESULTS_DIR"
echo "Compare KV transfer metrics:"
echo "  find $RESULTS_DIR -name metrics.json | sort"
echo "Compare vLLM request-level metrics:"
echo "  find $RESULTS_DIR -name vllm_bench.json | sort"
