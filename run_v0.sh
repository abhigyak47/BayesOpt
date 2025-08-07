#!/usr/bin/env bash
python -m baselines.dna       --n-workers 20
python -m baselines.mopta       --n-workers 20
python -m baselines.rbrock100 --n-workers 20
python -m baselines.rbrock300-100 --n-workers 20
python -m baselines.stybtang200 --n-workers 20
python -m baselines.rover --n-workers 20
