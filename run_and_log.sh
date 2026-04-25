#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <command>" >&2
  exit 1
fi

command="$1"
log_file="$PWD/log.txt"

printf '$ %s\n' "$command" >>"$log_file"
bash -c "$command" \
  > >(tee -i -a "$log_file") \
  2> >(tee -i -a "$log_file" >&2)
status=$?
printf '[exit_code=%d]\n' "$status" >>"$log_file"

exit "$status"
