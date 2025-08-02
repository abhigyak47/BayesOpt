#!/usr/bin/env bash
python -m baselines.ackley150 --n-workers 20
python -m baselines.ackley300-150 --n-workers 20
python -m baselines.hartmann6 --n-workers 20
python -m baselines.SVM --n-workers 20