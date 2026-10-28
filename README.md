# EA-GATSched: Graph-Attention Backfill Ordering Under Power Caps

Code, results and analysis for *"Graph-Attention Backfill Ordering Under Power Caps (EA-GATSched): A Trace-Driven Evaluation on ALCF Production Job Logs"* (Adimora and Sun, University of Kansas).

EA-GATSched is the energy-aware, graph-attention scheduler, which combines a graph-attention policy with energy-aware objectives. This repository evaluates its backfill-ordering component, GAT-BF.

## Policies
| Policy | Description |
|---|---|
| FCFS | head job only, no backfilling |
| EASY | EASY backfilling, candidates in arrival order |
| EASY-SJBF | same feasibility rule, shortest requested walltime first |
| Random-BF | same feasibility rule, random order |
| GAT-BF | same feasibility rule, order chosen by a dense GATv2 policy trained with REINFORCE against EASY |
| GAT-BF-WS | GAT-BF warm-started by 60 iterations of partial imitation of SJBF, then REINFORCE against SJBF (`GAT-BF-SJBFinit` in the result files) |

All reorderings keep EASY's head-job reservation. The imitation stage agrees with SJBF on only 34% to 73% of validation decisions, so GAT-BF-WS is not a copy of SJBF.

## Main findings (3 systems, 377,234 jobs, 74 held-out windows)
- FCFS waits 2.4 to 9.9 times longer than EASY.
- Reordering backfill candidates lowers pooled mean wait relative to EASY by 3% to 32% (learned variants) and by up to 36% (SJBF, no parameters).
- No learned policy is significantly better than SJBF. Four comparisons against SJBF are significant after Holm correction, all in SJBF's favour. GAT-BF trained from scratch is significantly worse than SJBF and than random ordering on uncapped Polaris.
- The one regime favouring learning is slowdown in congested, 85%-capped Polaris windows (exploratory, 14 windows).
- Energy per job differs by under 0.7% across reorderings. With always-on nodes, energy differs between schedules only through makespan.
- An earlier run with 20 Polaris test windows suggested a 50% wait reduction for GAT-BF; with all 45 windows it is 15.7%. Both are reported in the paper (sensitivity table); `archive_v1_first_run/` holds the first run.
- Simulated waits are not production predictions; use the simulator for relative comparisons.

## Layout
```
colab/ea_gatsched_v2.py     full pipeline (simulator, policies, training, evaluation)
colab/RUN_V2_INSTRUCTIONS.md
results_v2/                 outputs of the reported run (per-window results, summaries, run log)
analysis/analyze.py         pooled ratios, bootstrap CIs, Wilcoxon + Holm, strata, seeds
analysis/make_figures.py    figures in figs/
analysis/make_tables.py     LaTeX tables in tables/
figs/, tables/              generated figures and tables used by the manuscript
```

## Reproducing
1. Obtain the ALCF job logs (`DIM_JOB_COMPOSITE`: Polaris, Mira, Cooley) from the ALCF Data Catalog on IEEE DataPort (DOI 10.21227/bhfr-wx19). They are not redistributed here.
2. Run `colab/ea_gatsched_v2.py` in Google Colab (see `RUN_V2_INSTRUCTIONS.md`). The reported run took 10,043 s.
3. Copy the CSVs into `results_v2/`, then from the repository root run `python analysis/analyze.py`, `python analysis/make_figures.py`, `python analysis/make_tables.py`.

Key settings: windows of 1,500 jobs; chronological 60/10/30 split; all non-overlapping test windows (Polaris 45, Mira 10, Cooley 19); 120 training episodes; seeds 0, 1, 2; 85% power cap; P_busy = 149.48 W, P_idle = 69.86 W (GEOPM Theta telemetry, applied to all machines as an assumption). `analysis/geopm_validate_full.py` re-measures the levels on the complete GEOPM dataset (53.5M samples; mean 149.46 W, minimum 44.81 W) and reports the energy-counter consistency check; its output is Appendix B of the paper.

Note: `results_v2/paired_vs_easy.csv` (from the Colab run) averages per-window relative changes; the paper uses pooled ratios of means from `analysis/pairwise_pooled.csv`. The `CE nan` lines in the training log come from a window with no multi-candidate decisions and are not a training failure.

## Scope
The simulator omits priorities, fair-share, reservations, failures and topology. The power cap only reduces capacity (it never slows jobs). GAT-BF orders backfill candidates only. Mira has 10 test windows, so its intervals are wide.

## Acknowledgments
Supported in part by the US National Science Foundation under Grant #2441633. Traces provided by the Argonne Leadership Computing Facility.

## GEOPM power-level validation (Kaggle)

The node-power levels used by the simulator come from the Theta GEOPM power-balancer traces (DGEMM, 155 W balancer setting, 8,640 trace files, 4,320 nodes, 53.5M rows).

- Kaggle notebook: https://www.kaggle.com/code/kyrianadimora/geopm-validation
- Kaggle dataset: https://www.kaggle.com/datasets/kyrianadimora/alcf-geopm-power-balancer-155w
- Script (same code as the notebook): `analysis/geopm_validate_full.py`
- Outputs: `analysis/geopm_validation.json`, `analysis/geopm_data_quality.csv`, `analysis/geopm_per_node.csv` (the per-file CSV, feature-correlation CSV and LaTeX table are produced by the notebook and are not stored here)
- Reported in Appendix B of the paper. Runtime on Kaggle is about 100 minutes for the complete dataset.

Main results: mean package power 149.46 W (std 15.45 W, min 44.81 W, max 177.15 W); per-node mean std 5.70 W; 38.5% of samples above 155 W (the setting is an average budget); reported power agrees with the energy-counter power to a mean absolute error of 6.15 W (4.19%). The dataset has one benchmark at one cap and no idle periods, so it cannot validate the idle level or a sweep over caps.
