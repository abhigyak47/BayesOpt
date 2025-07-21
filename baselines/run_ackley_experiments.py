import os
import csv
import random
import torch
import numpy as np

from data import FuncAckley, BayesOptDataset
from . import BO_loop
from .BO_loop import BO_loop_GP

# Configurations
DIM = 10
NUM_INIT = 20
NUM_ITER = 50
BETA = 1.5
SEEDS = list(range(1, 11))
KERNELS = ["rbf", "matern32", "matern12", "rq"]  # rq = rational quadratic

RESULTS_CSV = "Ackley.csv"

def run_experiment():
    # Prepare CSV file with header
    fieldnames = ["seed", "kernel", "iteration", "best_obj_val"]
    with open(RESULTS_CSV, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for kernel in KERNELS:
        print(f"\n=== Running kernel: {kernel} ===")
        for seed in SEEDS:
            print(f"-- Seed: {seed} --")

            # Fix seeds for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.set_default_tensor_type(torch.DoubleTensor)

            # Instantiate function and dataset
            func = FuncAckley(DIM, maximize=True)
            dataset = BayesOptDataset(func, NUM_INIT, 'lhs', seed)

            try:
                # Run BO loop
                best_vals, _ = BO_loop_GP(
                    func_name="Ackley",
                    dataset=dataset,
                    seed=seed,
                    num_step=NUM_ITER,
                    beta=BETA,
                    if_ard=True,          # default ARD on
                    if_softplus=True,     # default softplus on
                    acqf_type="UCB",
                    set_ls=False,
                    kernel_type=kernel,
                )

                # Save results to CSV
                with open(RESULTS_CSV, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    for itr, val in enumerate(best_vals, start=1):
                        writer.writerow({
                            "seed": seed,
                            "kernel": kernel,
                            "iteration": itr,
                            "best_obj_val": val
                        })

                print(f"Seed {seed} with kernel {kernel} done.")

            except Exception as e:
                print(f"Error for seed {seed} kernel {kernel}: {e}")

if __name__ == "__main__":
    run_experiment()
