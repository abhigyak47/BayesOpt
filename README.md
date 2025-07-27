Run `python -m baselines.{script_name}` from the main directory without `.py`
Example for function `rosenbrock100`
Using 4 workers:
`python -m baselines.rbrock100 --n-workers 4`
Using all workers:
`python -m baselines.rbrock100` 


Results are stored in
`results/{function}/{kernel}/{function}_{kernel}_seed{seed}.csv`


Combining results:
For a specific function:
`python -m baselines.combine --function {function}`
Example:
`python -m baselines.combine --function rosenbrock100`
For all functions:
`python -m baselines.combine`

Final csv is stored in
`results/{function}`

