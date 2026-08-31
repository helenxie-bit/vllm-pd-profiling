#!/bin/bash

# =============================================================================
# vLLM Disaggregated Serving Script for Mooncake Connector
# =============================================================================
# This script demonstrates disaggregated prefill and decode serving using
# Mooncake Connector.
#
# Configuration can be customized via environment variables:
#   MODEL: Model to serve
#   PREFILL_GPUS: Comma-separated GPU IDs for prefill servers
#   DECODE_GPUS: Comma-separated GPU IDs for decode servers
#   PREFILL_PORTS: Comma-separated ports for prefill servers
#   BOOTSTRAP_PORTS: Bootstrap server port launched by prefill servers
#   DECODE_PORTS: Comma-separated ports for decode servers
#   PROXY_PORT: Proxy server port used to setup P/D disaggregated connection.
#   TIMEOUT_SECONDS: Server startup timeout
#   MOONCAKE_DEVICE_NAME: Optional RoCE/IB NIC whitelist (e.g. mlx5_0)
#   VLLM_MOONCAKE_PD_PROFILE: Set to 1 to enable PD stage timing marks
#   VLLM_MOONCAKE_PD_PROFILE_DIR: JSONL output directory
#   ENABLE_PREFIX_CACHING: Set to 1 to enable APC (disabled by default so
#                         profiling transfers the complete prompt KV cache)
# =============================================================================

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-8000}

PREFILL_GPUS=${PREFILL_GPUS:-0}
DECODE_GPUS=${DECODE_GPUS:-1}
PREFILL_PORTS=${PREFILL_PORTS:-8010}
BOOTSTRAP_PORTS=${BOOTSTRAP_PORTS:-8998}
DECODE_PORTS=${DECODE_PORTS:-8020}

MOONCAKE_DEVICE_NAME=${MOONCAKE_DEVICE_NAME:-}
VLLM_MOONCAKE_PD_PROFILE=${VLLM_MOONCAKE_PD_PROFILE:-0}
VLLM_MOONCAKE_PD_PROFILE_DIR=${VLLM_MOONCAKE_PD_PROFILE_DIR:-/tmp/vllm_pd_profile}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}

if [[ "$ENABLE_PREFIX_CACHING" == "1" || "$ENABLE_PREFIX_CACHING" == "true" ]]; then
    PREFIX_CACHING_ARG=--enable-prefix-caching
else
    PREFIX_CACHING_ARG=--no-enable-prefix-caching
fi

# Build kv_connector_extra_config JSON fragment.
build_extra_config() {
    local extras='"mooncake_protocol":"rdma"'
    if [[ -n "$MOONCAKE_DEVICE_NAME" ]]; then
        extras+=",\"device_name\":\"${MOONCAKE_DEVICE_NAME}\""
    fi
    echo "$extras"
}

KV_EXTRA=$(build_extra_config)

echo "Warning: Mooncake Connector support for vLLM v1 is experimental and subject to change."
echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS, Bootstrap Port:$BOOTSTRAP_PORTS"
echo "  Decode GPUs: $DECODE_GPUS, Ports: $DECODE_PORTS"
echo "  Proxy Port: $PROXY_PORT"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo "  Mooncake device_name: ${MOONCAKE_DEVICE_NAME:-<all>}"
echo "  PD profile: $VLLM_MOONCAKE_PD_PROFILE (dir=$VLLM_MOONCAKE_PD_PROFILE_DIR)"
echo "  Prefix caching: $ENABLE_PREFIX_CACHING"
echo ""

PIDS=()

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    local files=("mooncake_connector_proxy.py")
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Required file $file not found in $(pwd)"
            exit 1
        fi
    done
}

check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Please set it to your Hugging Face token."
        echo "Example: export HF_TOKEN=your_token_here"
        exit 1
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN is not a valid Hugging Face token. Please set it to your Hugging Face token."
        exit 1
    fi
    echo "HF_TOKEN is set and valid."
}

check_num_gpus() {
    # Check if the number of GPUs are >=2 via nvidia-smi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it via pip install $1."
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    pkill -9 -f "mooncake_connector_proxy.py"
    kill -- -$$            # negative PID  ==  "this whole process-group"
    wait                   # reap children so we don't leave zombies
    exit 0
}

wait_for_server() {
  local port=$1
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      echo "Server on port $port is ready."
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server on port $port"
      return 1
    fi

    sleep 1
  done
}

main() {
    check_required_files
    check_hf_token
    check_num_gpus
    ensure_python_library_installed vllm
    ensure_python_library_installed mooncake.engine

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    if [[ "$VLLM_MOONCAKE_PD_PROFILE" == "1" || "$VLLM_MOONCAKE_PD_PROFILE" == "true" ]]; then
        mkdir -p "$VLLM_MOONCAKE_PD_PROFILE_DIR"
        rm -f "$VLLM_MOONCAKE_PD_PROFILE_DIR"/*.jsonl
        export VLLM_MOONCAKE_PD_PROFILE=1
        export VLLM_MOONCAKE_PD_PROFILE_DIR
    fi

    echo "Launching disaggregated serving components..."
    echo "Please check the log files for detailed output:"
    echo "  - prefill*.log: Prefill server logs"
    echo "  - decode*.log: Decode server logs"
    echo "  - proxy.log: Proxy server log"

    # Parse GPU and port arrays
    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra BOOTSTRAP_PORT_ARRAY <<< "$BOOTSTRAP_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    proxy_args=()

    # =============================================================================
    # Launch Prefill Servers (X Producers)
    # =============================================================================
    echo ""
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local bootstrap_port=${BOOTSTRAP_PORT_ARRAY[$i]}

        echo "  Prefill server $((i+1)): GPU $gpu_id, Port $port, Bootstrap Port $bootstrap_port"
        VLLM_MOONCAKE_PD_PROFILE_ROLE=prefill \
        VLLM_MOONCAKE_BOOTSTRAP_PORT=$bootstrap_port CUDA_VISIBLE_DEVICES=$gpu_id \
        vllm serve "$MODEL" \
        --port "$port" \
        "$PREFIX_CACHING_ARG" \
        --kv-transfer-config \
        "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{${KV_EXTRA}}}" \
        > prefill$((i+1)).log 2>&1 &
        PIDS+=($!)
        proxy_args+=(--prefill "http://0.0.0.0:${port}" "$bootstrap_port")
    done

    # =============================================================================
    # Launch Decode Servers (Y Decoders)
    # =============================================================================
    echo ""
    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}

        echo "  Decode server $((i+1)): GPU $gpu_id, Port $port"
        VLLM_MOONCAKE_PD_PROFILE_ROLE=decode \
        CUDA_VISIBLE_DEVICES=$gpu_id vllm serve "$MODEL" \
        --port "$port" \
        "$PREFIX_CACHING_ARG" \
        --kv-transfer-config \
        "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{${KV_EXTRA}}}" \
        > decode$((i+1)).log 2>&1 &
        PIDS+=($!)
        proxy_args+=(--decode "http://0.0.0.0:${port}")
    done

    # =============================================================================
    # Launch Proxy Server
    # =============================================================================
    echo ""
    echo "Starting proxy server on port $PROXY_PORT..."
    VLLM_MOONCAKE_PD_PROFILE_ROLE=proxy \
    python3 mooncake_connector_proxy.py "${proxy_args[@]}" --port "$PROXY_PORT" > proxy.log 2>&1 &
    PIDS+=($!)

    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server "$port"; then
            echo "Failed to start server on port $port"
            cleanup
            # shellcheck disable=SC2317
            exit 1
        fi
    done

    echo ""
    echo "All servers are up. Starting benchmark..."

    # =============================================================================
    # Run Benchmark
    # =============================================================================
    vllm bench serve --port "$PROXY_PORT" --seed "$(date +%s)" \
        --backend vllm --model "$MODEL" \
        --dataset-name random --random-input-len 7500 --random-output-len 200 \
        --num-prompts 200 --burstiness 100 --request-rate 2 | tee benchmark.log

    if [[ "$VLLM_MOONCAKE_PD_PROFILE" == "1" || "$VLLM_MOONCAKE_PD_PROFILE" == "true" ]]; then
        echo ""
        echo "Analyzing PD profile marks in $VLLM_MOONCAKE_PD_PROFILE_DIR ..."
        python3 analyze_pd_profile.py --dir "$VLLM_MOONCAKE_PD_PROFILE_DIR" || true
    fi

    echo "Benchmarking done. Cleaning up..."

    cleanup
}

main
