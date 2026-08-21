#!/usr/bin/env bash
set -uex
PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONPATH

python -m slim_profiler.profiler \
    --trace-pid "$$" \
    --dst-tsv "profile.${SLURM_JOB_NAME:-bash}.${SLURM_JOB_ID:-$$}" \
    --interval 1 &>>"profile.${SLURM_JOB_NAME:-bash}.${SLURM_JOB_ID:-$$}.log" &
TRACE_PID="$!"

stress-ng -t 10 --zlib 10

kill -s TERM "${TRACE_PID}"
wait "${TRACE_PID}" || true
python -m slim_profiler.plot --dst-tsv "profile.${SLURM_JOB_NAME:-bash}.${SLURM_JOB_ID:-$$}"
