#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

declare -a requested_modules=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --module)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --module" >&2
        exit 1
      fi
      requested_modules+=("$2")
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

gomodules=()
if [ "${#requested_modules[@]}" -gt 0 ]; then
  for module in "${requested_modules[@]}"; do
    normalized="${module#./}"
    normalized="./${normalized}"
    if [ ! -f "${normalized}" ]; then
      echo "module file not found: ${module}" >&2
      exit 1
    fi
    gomodules+=("${normalized}")
  done
else
  while IFS= read -r mod; do
    if head -n 5 "${mod}" | grep -q "DO NOT USE!"; then
      continue
    fi
    gomodules+=("${mod}")
  done < <(find . -name go.mod | sort)
fi

if [ "${#gomodules[@]}" -eq 0 ]; then
  echo "no go modules found"
  exit 1
fi

for mod in "${gomodules[@]}"; do
  mod_dir="$(dirname "${mod}")"
  rel_dir="${mod_dir#./}"
  if [ "${rel_dir}" = "" ] || [ "${rel_dir}" = "." ]; then
    readable_name="root"
  else
    readable_name="${rel_dir}"
  fi
  safe_name="${readable_name//\//_}"
  safe_name="${safe_name//./_}"
  echo "::group::Testing ${safe_name}"
  echo "module dir: ${readable_name}"
  (
    cd "${mod_dir}"
    packages="$(go list ./... 2>/dev/null || true)"
    if [ -z "${packages}" ]; then
      echo "no packages to test in ${readable_name}, skipping"
      exit 0
    fi
    go test ./...
  )
  echo "::endgroup::"
done
