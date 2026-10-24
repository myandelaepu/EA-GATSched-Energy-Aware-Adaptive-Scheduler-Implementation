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

Key settings: windows of 1,500 jobs; chronological 60/10/30 split; all non-overlapping test windows (Polaris 45, Mira 10, Cooley 19); 120 training episodes; seeds 0, 1, 2; 85% power cap; P_busy = 149.48 W, P_idle = 69.86 W (GEOPM Theta telemetry, applied to all machines as an assumption).

Note: `results_v2/paired_vs_easy.csv` (from the Colab run) averages per-window relative changes; the paper uses pooled ratios of means from `analysis/pairwise_pooled.csv`. The `CE nan` lines in the training log come from a window with no multi-candidate decisions and are not a training failure.

## Scope
The simulator omits priorities, fair-share, reservations, failures and topology. The power cap only reduces capacity (it never slows jobs). GAT-BF orders backfill candidates only. Mira has 10 test windows, so its intervals are wide.

## Overlap with earlier work
The same traces were used in Adimora et al., Future Generation Computer Systems 186 (2027) 108763, doi:10.1016/j.future.2026.108763, whose protocol and policy differ from this controlled study.

## Acknowledgments
Supported in part by the US National Science Foundation under Grant #2441633. Traces provided by the Argonne Leadership Computing Facility.
