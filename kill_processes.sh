# force-kill python* started ≤ 300s ago
ps -eo pid=,etimes=,comm= \
| awk '$3 ~ /^python([0-9.]*)?$/ && $2 <= 300 {print $1}' \
| xargs -r sudo kill -9 --
