# Data contract

Raw factor source files are intentionally not redistributed in this repository.

For replication, prepare a monthly CSV with at least:

```text
date,MKT,SMB,HML,RMW,CMA,MOM
```

Conventions used in the research:

- one row per month;
- no duplicate dates;
- factor returns in a single consistent unit (percent or decimal, but not mixed);
- no forward-filled factor observations;
- sort strictly by date before walk-forward estimation;
- standardize using **training-window statistics only** in out-of-sample experiments.

Recommended source: the Kenneth R. French Data Library for Fama-French factors and momentum. Verify the exact vintage used in any publication-grade replication, because historical factor files can be revised.
