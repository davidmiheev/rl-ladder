#!/bin/bash
# Sequential within a stream; streams run in parallel.
cd /home/user/rl-ladder
stream="$1"; shift
for spec in "$@"; do
  IFS='|' read -r strat sym days <<< "$spec"
  echo "### $strat $sym $days  $(date +%T)"
  .venv/bin/python experiments/c6_strategy_zoo/run_baseline.py \
     --strategy "$strat" --symbol "$sym" --days "$days" --tag "base-$strat-$sym" || echo "FAILED $spec"
done
echo "### stream $stream done $(date +%T)"
