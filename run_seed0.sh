#!/usr/bin/env bash
# python -m baselines.rover --n-workers 20
# python -m baselines.dna       --n-workers 20
# python -m baselines.mopta       --n-workers 20
# python -m baselines.rbrock100 --n-workers 20
# python -m baselines.rbrock300-100 --n-workers 20
# python -m baselines.stybtang200 --n-workers 20

python -m baselines.ackleyKRY150 --n-workers 20
python -m baselines.ackleyKRY300-150 --n-workers 20
# python -m baselines.hartmann6 --n-workers 20
# python -m baselines.SVM --n-workers 20