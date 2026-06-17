#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_dataset="$script_dir/../summary/data/longmemeval-cleaned/longmemeval_s_cleaned.json"
default_output="$script_dir/results/lme_single_session_user_70"

benchmark="auto,adk,agno"
dataset="$default_dataset"
output="$default_output"
model="${LLM_NAME:-gpt-4o-mini}"
eval_model="${EVAL_MODEL_NAME:-}"
embed_model="${EMBED_MODEL_NAME:-text-embedding-3-small}"
max_tasks="70"
question_types="single-session-user"
python_bin="${PYTHON_BIN:-}"
go_memory_backend="${GO_MEMORY_BACKEND:-pgvector}"
table_suffix="${TABLE_SUFFIX:-}"
timeout_value="${TIMEOUT:-}"
llm_judge="0"
dry_run="0"

usage() {
  cat <<'EOF'
Usage: ./run_lme_benchmarks.sh [options]

Run selected 70-case LongMemEval single-session-user benchmarks in order.

Options:
  --benchmark LIST          Comma-separated benchmarks in execution order: auto,adk,agno
                            Default: auto,adk,agno
  --dataset PATH            LongMemEval cleaned dataset path
  --output DIR              Output root directory
  --model NAME              LLM model name. Default: $LLM_NAME, then gpt-4o-mini
  --eval-model NAME         Optional LLM judge model name
  --embed-model NAME        Embedding model for Go auto vector memory. Default: $EMBED_MODEL_NAME or text-embedding-3-small
  --max-tasks N             Number of cases to run. Default: 70
  --question-types LIST     LongMemEval question types. Default: single-session-user
  --python PATH             Python executable for ADK/Agno. Default: first python3.12..python3.7, then python3
  --go-memory-backend NAME  Go auto memory backend: pgvector, sqlitevec, sqlite, mysql, inmemory. Default: pgvector
  --table-suffix SUFFIX     Optional Go DB table suffix. Default: _lme_ssu_70_auto
  --timeout DURATION        Optional timeout duration, for example 4h or 30m
  --llm-judge               Enable LLM judge for Python benchmarks
  --dry-run                 Print commands without running them
  -h, --help                Show this help

Examples:
  ./run_lme_benchmarks.sh --benchmark adk,auto
  ./run_lme_benchmarks.sh --benchmark auto,adk --dry-run
  ./run_lme_benchmarks.sh --benchmark adk,agno --python /usr/bin/python3.8
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
      benchmark="${2:?missing value for --benchmark}"
      shift 2
      ;;
    --dataset)
      dataset="${2:?missing value for --dataset}"
      shift 2
      ;;
    --output)
      output="${2:?missing value for --output}"
      shift 2
      ;;
    --model)
      model="${2:?missing value for --model}"
      shift 2
      ;;
    --eval-model)
      eval_model="${2:?missing value for --eval-model}"
      shift 2
      ;;
    --embed-model)
      embed_model="${2:?missing value for --embed-model}"
      shift 2
      ;;
    --max-tasks)
      max_tasks="${2:?missing value for --max-tasks}"
      shift 2
      ;;
    --question-types)
      question_types="${2:?missing value for --question-types}"
      shift 2
      ;;
    --python)
      python_bin="${2:?missing value for --python}"
      shift 2
      ;;
    --go-memory-backend)
      go_memory_backend="${2:?missing value for --go-memory-backend}"
      shift 2
      ;;
    --table-suffix)
      table_suffix="${2:?missing value for --table-suffix}"
      shift 2
      ;;
    --timeout)
      timeout_value="${2:?missing value for --timeout}"
      shift 2
      ;;
    --llm-judge)
      llm_judge="1"
      shift
      ;;
    --dry-run)
      dry_run="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$(pwd)/$path"
  fi
}

select_python() {
  if [[ -n "$python_bin" ]]; then
    printf '%s\n' "$python_bin"
    return
  fi
  local candidate
  for candidate in python3.12 python3.11 python3.10 python3.9 python3.8 python3.7 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$(command -v "$candidate")"
      return
    fi
  done
  echo "python3"
}

python_version_ok() {
  local py="$1"
  "$py" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 7) else 1)
PY
}

normalize_env() {
  if [[ -z "${OPENAI_API_KEY:-}" && -n "${LLM_API_KEY:-}" ]]; then
    export OPENAI_API_KEY="$LLM_API_KEY"
  fi
  if [[ -z "${LLM_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
    export LLM_API_KEY="$OPENAI_API_KEY"
  fi
  if [[ -z "${OPENAI_EMBEDDING_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
    export OPENAI_EMBEDDING_API_KEY="$OPENAI_API_KEY"
  fi
}

ensure_go_cgo_flags() {
  local sqlite_innocuous_flag="-DSQLITE_INNOCUOUS=0x000200000"
  if [[ " ${CGO_CFLAGS:-} " != *" $sqlite_innocuous_flag "* ]]; then
    export CGO_CFLAGS="${CGO_CFLAGS:+$CGO_CFLAGS }$sqlite_innocuous_flag"
  fi
}

validate_benchmark() {
  case "$1" in
    auto|adk|agno)
      ;;
    *)
      echo "Unsupported benchmark: $1. Supported: auto,adk,agno" >&2
      exit 2
      ;;
  esac
}

ensure_env_for_benchmark() {
  local name="$1"
  if [[ -z "${OPENAI_API_KEY:-}" && -z "${LLM_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY or LLM_API_KEY is required" >&2
    exit 2
  fi
  if [[ "$name" == "auto" && ( "$go_memory_backend" == "pgvector" || "$go_memory_backend" == "sqlitevec" ) ]]; then
    if [[ -z "${OPENAI_EMBEDDING_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" && -z "${LLM_API_KEY:-}" ]]; then
      echo "OPENAI_EMBEDDING_API_KEY, OPENAI_API_KEY, or LLM_API_KEY is required for vector memory backends" >&2
      exit 2
    fi
    if [[ "$go_memory_backend" == "pgvector" && -z "${PGVECTOR_DSN:-}" ]]; then
      echo "PGVECTOR_DSN is required for Go auto pgvector" >&2
      exit 2
    fi
  fi
}

run_cmd() {
  local cwd="$1"
  shift
  local cmd=("$@")
  if [[ -n "$timeout_value" ]]; then
    cmd=(timeout "$timeout_value" "${cmd[@]}")
  fi

  printf '\n>>> cd %q &&' "$cwd"
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "$dry_run" == "1" ]]; then
    return
  fi
  (cd "$cwd" && "${cmd[@]}")
}

run_python_benchmark() {
  local name="$1"
  local workdir output_subdir scenario
  case "$name" in
    adk)
      workdir="$script_dir/adk-python-impl"
      output_subdir="adk_python"
      scenario="native_memory"
      ;;
    agno)
      workdir="$script_dir/agno-impl"
      output_subdir="agno_python"
      scenario="native_memory"
      ;;
    *)
      echo "Internal error: unsupported Python benchmark $name" >&2
      exit 2
      ;;
  esac

  local cmd=(
    "$selected_python"
    main.py
    --dataset-format longmemeval
    --dataset "$dataset_abs"
    --scenario "$scenario"
    --question-types "$question_types"
    --max-tasks "$max_tasks"
    --model "$model"
    --output "$output_abs/$output_subdir"
  )
  if [[ -n "$eval_model" ]]; then
    cmd+=(--eval-model "$eval_model")
  fi
  if [[ "$llm_judge" == "1" ]]; then
    cmd+=(--llm-judge)
  fi
  run_cmd "$workdir" "${cmd[@]}"
}

run_go_auto() {
  ensure_go_cgo_flags

  local suffix="$table_suffix"
  if [[ -z "$suffix" ]]; then
    suffix="_lme_ssu_70_auto"
  fi

  local cmd=(
    go run .
    -dataset-format longmemeval
    -dataset "$dataset_abs"
    -scenario auto
    -memory-backend "$go_memory_backend"
    -output "$output_abs"
    -model "$model"
    -embed-model "$embed_model"
    -max-tasks "$max_tasks"
    -lme-question-types "$question_types"
    -table-suffix "$suffix"
    -resume
  )
  if [[ -n "$eval_model" ]]; then
    cmd+=(-eval-model "$eval_model")
  fi
  run_cmd "$script_dir/trpc-agent-go-impl" "${cmd[@]}"
}

dataset_abs="$(abs_path "$dataset")"
output_abs="$(abs_path "$output")"

if [[ ! -f "$dataset_abs" ]]; then
  echo "dataset not found: $dataset_abs" >&2
  exit 2
fi
mkdir -p "$output_abs"

selected_python="$(select_python)"
if ! command -v "$selected_python" >/dev/null 2>&1; then
  echo "Python executable not found: $selected_python" >&2
  exit 2
fi
if ! python_version_ok "$selected_python"; then
  echo "Python executable must be >= 3.7 for ADK/Agno: $selected_python" >&2
  echo "Please install a newer Python or pass --python /path/to/python3.8+." >&2
  exit 2
fi

normalize_env

IFS=',' read -r -a benchmarks <<< "$benchmark"
ordered_benchmarks=()
for raw_name in "${benchmarks[@]}"; do
  name="${raw_name//[[:space:]]/}"
  [[ -z "$name" ]] && continue
  validate_benchmark "$name"
  ordered_benchmarks+=("$name")
done

if [[ ${#ordered_benchmarks[@]} -eq 0 ]]; then
  echo "--benchmark must include at least one benchmark" >&2
  exit 2
fi

printf 'Running benchmarks in order: '
for idx in "${!ordered_benchmarks[@]}"; do
  if [[ "$idx" != "0" ]]; then
    printf ', '
  fi
  printf '%s' "${ordered_benchmarks[$idx]}"
done
printf ' | cases=%s | question_types=%s | python=%s\n' "$max_tasks" "$question_types" "$selected_python"

for name in "${ordered_benchmarks[@]}"; do
  ensure_env_for_benchmark "$name"
  case "$name" in
    auto)
      run_go_auto
      ;;
    adk|agno)
      run_python_benchmark "$name"
      ;;
  esac
done

printf '\nAll requested benchmarks completed. Output root: %s\n' "$output_abs"
