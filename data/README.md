# Data contract

Raw factor source files are intentionally not redistributed in this repository.

For the core replication pipeline, prepare a monthly CSV with:

```text
date,MKT,SMB,HML,RMW,CMA,MOM
```

Conventions:

- one row per month;
- no duplicate dates;
- factor returns in a single consistent unit (percent or decimal, but not mixed);
- no forward-filled factor observations;
- finite numeric observations only;
- sort strictly by date;
- for walk-forward/OOS experiments, standardize using **training-window statistics only**.

Recommended source: the Kenneth R. French Data Library for Fama-French factors and momentum. Record the exact download date/vintage in publication-grade work because historical files can be revised.

Run the core replication with:

```bash
python scripts/run_research.py /path/to/factors.csv --output results/reproduced
```
