#!/usr/bin/env bash
set -euo pipefail

env_file="${LME_ENV_FILE:-../../../autotest/workflow/.env.local}"
if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$env_file"
  set +a
fi

timeout_duration="${LME_GUARD_TIMEOUT:-4h}"
kill_after="${LME_GUARD_KILL_AFTER:-120s}"
model_name="${LLM_NAME:-gpt-4o-mini}"
embed_model="${EMBED_MODEL_NAME:-text-embedding-ada-002}"
scenario="${LME_SCENARIO:-auto}"
table_suffix="${LME_TABLE_SUFFIX:-_lme_guarded_20260611}"
question_types="${LME_QUESTION_TYPES:-single-session-user}"
graph_mr_memory_backend="${LME_GRAPH_MR_MEMORY_BACKEND:-inmemory}"
output_dir="${LME_OUTPUT_DIR:-../results/data_longmemeval_guarded}"
dataset_path="${LME_DATASET_PATH:-../../summary/data/longmemeval-cleaned/longmemeval_s_cleaned.json}"
binary_path="${LME_BINARY_PATH:-/tmp/lme-memory-bench}"
log_path="${LME_RUN_LOG:-$output_dir/run_guarded.log}"
default_pgvector_dsn="postgres://postgres@localhost:55432/trpc_agent_summary_bench?sslmode=disable"

mkdir -p "$output_dir"
exec > >(tee -a "$log_path") 2>&1

echo "[$(date -Is)] LongMemEval guarded run starting"
echo "timeout=$timeout_duration kill_after=$kill_after model=$model_name embed=$embed_model scenario=$scenario"
echo "output=$output_dir table_suffix=$table_suffix question_types=${question_types:-all} graph_mr_memory_backend=$graph_mr_memory_backend"

if [[ -z "${LLM_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export LLM_API_KEY="$OPENAI_API_KEY"
fi
if [[ -z "${OPENAI_EMBEDDING_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_EMBEDDING_API_KEY="$OPENAI_API_KEY"
fi
if [[ -z "${LLM_API_KEY:-}" ]]; then
  echo "LLM_API_KEY is missing; refusing to run"
  exit 2
fi
if [[ -z "${PGVECTOR_DSN:-}" ]]; then
  export PGVECTOR_DSN="$default_pgvector_dsn"
fi
if [[ -z "${PGVECTOR_DSN:-}" ]]; then
  echo "PGVECTOR_DSN is missing; refusing to run"
  exit 2
fi
if command -v pg_isready >/dev/null 2>&1; then
  echo "[$(date -Is)] Checking pgvector DSN"
  if ! pg_isready -d "$PGVECTOR_DSN"; then
    echo "PGVECTOR_DSN is not reachable; refusing to run"
    exit 2
  fi
fi
if command -v curl >/dev/null 2>&1; then
  embedding_base="${OPENAI_EMBEDDING_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
  embedding_base="${embedding_base%/}"
  embedding_key="${OPENAI_EMBEDDING_API_KEY:-${OPENAI_API_KEY:-${LLM_API_KEY:-}}}"
  preflight_body="/tmp/lme_embedding_preflight_$$.json"
  echo "[$(date -Is)] Checking embedding endpoint"
  http_code="$(curl -sS -o "$preflight_body" -w '%{http_code}' \
    --connect-timeout 10 \
    --max-time 30 \
    -H "Authorization: Bearer $embedding_key" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$embed_model\",\"input\":\"LongMemEval embedding preflight\"}" \
    "$embedding_base/embeddings" || true)"
  rm -f "$preflight_body"
  if [[ "$http_code" != 2* ]]; then
    echo "Embedding endpoint preflight failed with HTTP $http_code; set OPENAI_EMBEDDING_BASE_URL/OPENAI_EMBEDDING_API_KEY before running"
    exit 2
  fi
fi

export GOCACHE="${GOCACHE:-/tmp/go-build-cache}"
export CGO_CFLAGS="${CGO_CFLAGS:--DSQLITE_INNOCUOUS=0x000200000}"

echo "[$(date -Is)] Building benchmark binary"
go build -o "$binary_path" .

echo "[$(date -Is)] Running benchmark under timeout"
args=(
  -dataset "$dataset_path"
  -dataset-format longmemeval
  -scenario "$scenario"
  -memory-backend pgvector
  -output "$output_dir"
  -model "$model_name"
  -embed-model "$embed_model"
  -table-suffix "$table_suffix"
  -lme-graph-mr-memory-backend "$graph_mr_memory_backend"
  -resume
)
if [[ -n "$question_types" ]]; then
  args+=(-lme-question-types "$question_types")
fi
set +e
timeout --kill-after="$kill_after" "$timeout_duration" "$binary_path" "${args[@]}"
status=$?
set -e

case "$status" in
  0)
    echo "[$(date -Is)] Benchmark completed"
    ;;
  124)
    echo "[$(date -Is)] Benchmark stopped by timeout after $timeout_duration"
    ;;
  137)
    echo "[$(date -Is)] Benchmark killed after timeout grace period $kill_after"
    ;;
  *)
    echo "[$(date -Is)] Benchmark exited with status $status"
    ;;
esac

exit "$status"
