# rbrock50.py
import os
import csv
import random
import time
import argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
torch.set_default_dtype(torch.float64)
import numpy as np

from data import FuncRosenbrock_V1, BayesOptDataset
from .BO_loop import BO_loop_GP

# ----- globals you likely want to tweak -----
DIM        = 100
NUM_INIT   = 20
NUM_ITER   = 20
BETA       = 1.5
SEEDS      = list(range(1, 11))
#KERNELS = ["rbf", "mat12", "mat52", "lin*mat52", "gcauchy", "poly2", "poly2*mat52", "mat52+const","rq"]
KERNELS = ["poly2"]
RESULTS_CSV = "rbrock1002.csv"

DEVICE = torch.device("cpu")

def run_one(kernel: str, seed: int, num_iter: int, beta: float, csv_path:str):

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

    #Write results asap for this seed
    fieldnames = ["seed", "kernel", "iteration", "best_obj_val"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()     
        for itr, val in enumerate(best_vals, 1):
            writer.writerow({
                "seed": seed,
                "kernel": kernel,
                "iteration": itr,
                "best_obj_val": float(val),
            })
    
    return len(best_vals)

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

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "kernel", "iteration", "best_obj_val"])
        writer.writeheader()

    tasks = list(product(KERNELS, SEEDS))
    max_workers = min(args.n_workers, len(tasks))
    print(f"Using {max_workers} workers for {len(tasks)} jobs")

    start = time.time()

    # safer with torch
    torch.multiprocessing.set_start_method("spawn", force=True)
    total_rows = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_one, k, s, args.num_iter, BETA, args.csv): (k, s)
            for k, s in tasks
        }
        for fut in as_completed(futures):
            k, s = futures[fut]
            try:
                rows_written = fut.result()
                total_rows += rows_written
                print(f"Completed kernel={k} seed={s} ({rows_written} rows)")
            except Exception as e:
                print(f"[ERROR] kernel={k} seed={s}: {e}")

    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.2f} min. Wrote {total_rows} rows to {args.csv}")
    if os.path.exists(args.csv):
        with open(args.csv, newline="") as f:
            reader = csv.DictReader(f)
            sorted_rows = sorted(reader, key=lambda r: (r["kernel"], int(r["seed"]), int(r["iteration"])))

        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "kernel", "iteration", "best_obj_val"])
            writer.writeheader()
            writer.writerows(sorted_rows)


if __name__ == "__main__":
    main()
