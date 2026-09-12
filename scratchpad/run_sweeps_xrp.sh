#!/bin/bash
cd /home/user/rl-ladder
echo "### xrp half-spread floor $(date +%T)"
.venv/bin/python experiments/c6_strategy_zoo/sweep_spread.py --strategy glft --param half_spread_bp \
    --symbol xrpusdc --days 4 --write || echo FAILED-hs
echo "### xrp grid depth $(date +%T)"
.venv/bin/python experiments/c6_strategy_zoo/sweep_spread.py --strategy glft --param grid_num \
    --symbol xrpusdc --days 4 || echo FAILED-grid
echo "### xrp basis grid depth $(date +%T)"
.venv/bin/python experiments/c6_strategy_zoo/sweep_spread.py --strategy basis --param grid_num \
    --symbol xrpusdc --days 4 || echo FAILED-grid2
echo "### done $(date +%T)"
