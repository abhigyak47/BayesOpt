import os
import csv
import random
import time  # ← import time for timing
import torch
import numpy as np

from data import FuncRosenbrock_V1, BayesOptDataset
from . import BO_loop
from .BO_loop import BO_loop_GP  

# Configurations
DIM = 100
NUM_INIT = 20
NUM_ITER = 400
BETA = 1.5
SEEDS = list(range(1, 11))
KERNELS = ["rbf", "mat12", "mat52", "lin*mat52", "gcauchy", "poly2", "poly2*mat52", "mat52+const"]
RESULTS_CSV = "rbrock100.csv"

def run_experiment():
    # Prepare CSV file with header
    fieldnames = ["seed", "kernel", "iteration", "best_obj_val"]
    with open(RESULTS_CSV, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for kernel in KERNELS:
        print(f"\n=== Running kernel: {kernel} ===")
        start_time = time.time()  # Start timer for this kernel

        for seed in SEEDS:
            print(f"-- Seed: {seed} --")

            # Fix seeds for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.set_default_tensor_type(torch.DoubleTensor)

            # Instantiate function and dataset
            func = FuncRosenbrock_V1(DIM, maximize=True)
            dataset = BayesOptDataset(func, NUM_INIT, 'lhs', seed)

            try:
                # Run BO loop
                best_vals, _ = BO_loop_GP(
                    func_name="Rosenbrock",
                    dataset=dataset,
                    seed=seed,
                    num_step=NUM_ITER,
                    beta=BETA,
                    if_ard=True,
                    if_softplus=True,
                    acqf_type="UCB",
                    set_ls=False,
                    kernel_type=kernel,  # must match string in KERNEL_DEFAULTS
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
                print(f"Error for seed {seed}, kernel {kernel}: {e}")

        # End timer and print duration
        elapsed = time.time() - start_time
        print(f"=== Kernel {kernel} completed in {elapsed:.2f} seconds ({elapsed/60:.2f} min) ===")

if __name__ == "__main__":
    run_experiment()
