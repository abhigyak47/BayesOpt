# rbrock50.py
import os
import csv
import random
import time
import argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np

from data import FuncRosenbrock_V1, BayesOptDataset
from .BO_loop import BO_loop_GP

# ----- globals you likely want to tweak -----
DIM        = 50
NUM_INIT   = 20
NUM_ITER   = 400
BETA       = 1.5
SEEDS      = list(range(1, 11))
KERNELS = ["rbf", "mat12", "mat52", "lin*mat52", "gcauchy", "poly2", "poly2*mat52", "mat52+const"]
RESULTS_CSV = "rbrock50.csv"

DEVICE = torch.device("cpu")

def run_one(kernel: str, seed: int, num_iter: int, beta: float):
    # keep each worker single-threaded to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    func = FuncRosenbrock_V1(DIM, maximize=True)
    dataset = BayesOptDataset(func, NUM_INIT, 'lhs', seed)

    best_vals, _ = BO_loop_GP(
        func_name="Rosenbrock",
        dataset=dataset,
        seed=seed,
        num_step=num_iter,
        beta=beta,
        if_ard=True,
        if_softplus=True,
        acqf_type="UCB",
        set_ls=False,
        kernel_type=kernel,
        device=DEVICE
    )

    rows = []
    for itr, val in enumerate(best_vals, 1):
        rows.append({
            "kernel": kernel,
            "seed": seed,
            "iteration": itr,
            "best_obj_val": float(val),
        })
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-workers", type=int, default=os.cpu_count()-1 or 1,
                   help="max parallel processes (default: os.cpu_count()-1)")
    p.add_argument("--csv", type=str, default=RESULTS_CSV,
                   help="output CSV path")
    p.add_argument("--num-iter", type=int, default=NUM_ITER,
                   help="BO iterations per (kernel, seed)")
    return p.parse_args()


def main():
    args = parse_args()

    tasks = list(product(KERNELS, SEEDS))
    max_workers = min(args.n_workers, len(tasks))
    print(f"Using {max_workers} workers for {len(tasks)} jobs")

    start = time.time()

    # safer with torch
    torch.multiprocessing.set_start_method("spawn", force=True)

    all_rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_one, k, s, args.num_iter, BETA): (k, s)
            for k, s in tasks
        }
        for fut in as_completed(futures):
            k, s = futures[fut]
            try:
                all_rows.extend(fut.result())
            except Exception as e:
                print(f"[ERROR] kernel={k} seed={s}: {e}")

    # deterministic order: kernel, seed, iteration
    all_rows.sort(key=lambda r: (r["kernel"], r["seed"], r["iteration"]))

    fieldnames = ["seed", "kernel", "iteration", "best_obj_val"]
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            # re-order to match header
            writer.writerow({
                "seed": r["seed"],
                "kernel": r["kernel"],
                "iteration": r["iteration"],
                "best_obj_val": r["best_obj_val"],
            })

    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.2f} min. Wrote {len(all_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
