import os
import glob
import argparse
import pandas as pd

def combine_function_results(root_dir: str, func_name: str):
    func_path = os.path.join(root_dir, func_name)
    if not os.path.exists(func_path):
        print(f"[!] Function folder not found: {func_path}")
        return

    pattern = os.path.join(func_path, "*", f"{func_name}_*_seed*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"[!] No result files found for function '{func_name}' in {func_path}")
        return

    all_dfs = []
    for filepath in csv_files:
        try:
            kernel = os.path.basename(os.path.dirname(filepath))
            filename = os.path.basename(filepath)
            seed_str = filename.split("_")[-1].replace("seed", "").replace(".csv", "")
            seed = int(seed_str)

            df = pd.read_csv(filepath)
            df["function"] = func_name
            df["kernel"] = kernel
            df["seed"] = seed
            all_dfs.append(df)
        except Exception as e:
            print(f"[ERROR] Failed to process {filepath}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.sort_values(by=["kernel", "seed", "iteration"], inplace=True)

        # Ensure consistent column order
        cols = ["seed", "kernel", "iteration", "best_obj_val", "function"]
        ordered_cols = [col for col in cols if col in combined.columns]
        combined = combined[ordered_cols]

        out_path = os.path.join(func_path, f"{func_name}_results.csv")
        combined.to_csv(out_path, index=False)
        print(f"[✓] Wrote: {out_path}")
    else:
        print(f"[!] No valid CSVs parsed for function '{func_name}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Root results directory (default: results/)")
    parser.add_argument("--function", type=str, default=None,
                        help="Function to combine (default: all functions in results/)")
    args = parser.parse_args()

    if args.function:
        combine_function_results(args.results_dir, args.function)
    else:
        # Combine for all functions
        function_dirs = [
            f for f in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, f))
        ]
        for func_name in function_dirs:
            combine_function_results(args.results_dir, func_name)

if __name__ == "__main__":
    main()
