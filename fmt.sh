#!/usr/bin/env bash
# shellcheck disable=SC2086
set -ue

if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY RUN ON"
fi

git ls-files |
    grep -v '.idea/' |
    grep '\.py$' |
    while read -r line; do
        if [ -e "${line}" ]; then
            {
                echo BLACK "${line}"
                if [ -n "${DRY_RUN:-}" ]; then
                    true
                else
                    black \
                        --line-length 120 --target-version py311 --quiet "${line}" &>/dev/null
                fi
            } &
        fi
    done
wait
