# EA-GATSched: Graph-Attention Backfill Ordering Under Power Caps

Code, results and analysis for the manuscript *"Graph-Attention Backfill Ordering Under Power Caps (EA-GATSched): A Trace-Driven Evaluation on ALCF Production Job Logs"* by Kyrian Chinemeze Adimora and Hongyang Sun, University of Kansas.

EA-GATSched is an energy-aware, graph-attention scheduler that combines a graph-attention policy with energy-aware objectives. This repository evaluates its backfill-ordering component, **GAT-BF**, under node-power models and power caps, on production job logs from three Argonne Leadership Computing Facility (ALCF) systems.

## What the study asks

A batch scheduler repeatedly decides which waiting job runs next. With EASY backfilling, the job at the head of the queue gets a reservation, and other jobs may start early only if they cannot delay it. Several jobs often qualify, and EASY takes them in arrival order. The study asks three questions about that one choice:

1. How much does the order of backfill candidates matter?
2. Does a learned ordering beat a zero-parameter rule such as shortest-job-backfilled-first (SJBF)?
3. Does a better ordering save energy?

It is examined at four scales: the single backfill decision (at most 32 candidates), the 1,500-job window, the system (126 to 49,152 nodes, with and without an 85% power cap), and the workload (377,234 jobs).

## Policies

| Policy | Description |
|---|---|
| FCFS | head job only, no backfilling |
| EASY | EASY backfilling, candidates in arrival order |
| EASY-SJBF | same feasibility rule, shortest requested walltime first |
| Random-BF | same feasibility rule, random order |
| GAT-BF | same feasibility rule, order chosen by a dense GATv2 policy trained with REINFORCE against EASY |
| GAT-BF-WS | GAT-BF warm-started by 60 iterations of partial imitation of SJBF, then REINFORCE against SJBF (`GAT-BF-SJBFinit` in the result files) |

All reorderings keep EASY's head-job reservation, so no job is delayed beyond what EASY would allow. The imitation stage agrees with SJBF on only 34% to 73% of validation decisions, so GAT-BF-WS is not a copy of SJBF.

## Main findings (3 systems, 377,234 jobs, 74 held-out windows)

- FCFS waits 2.4 to 9.9 times longer than EASY.
- Reordering backfill candidates lowers pooled mean wait relative to EASY by 3% to 32% (learned variants) and by up to 36% (SJBF, which has no parameters).
- No learned policy is significantly better than SJBF. Four comparisons against SJBF are significant after Holm correction, and all four favour SJBF. GAT-BF trained from scratch is significantly worse than SJBF and than random ordering on uncapped Polaris.
- The one regime favouring learning is slowdown in congested, 85%-capped Polaris windows (exploratory, 14 windows).
- Energy per job differs by under 0.7% across reorderings. With always-on nodes, energy differs between schedules only through makespan (Proposition 2 in the paper), and a power cap is exactly a node-count limit (Proposition 1).
- Utilization and throughput stay within 1.4% of EASY for every reordering.
- An earlier run with 20 Polaris test windows suggested a 50% wait reduction for GAT-BF; with all 45 windows it is 15.7%. Both are reported in the paper as a sensitivity check, and `archive_v1_first_run/` holds the first run.
- Simulated waits are two to six times lower than production waits. Use the simulator for relative comparisons only.

## Repository layout

```
colab/ea_gatsched_v2.py              full pipeline: simulator, policies, training, evaluation (the reported run)
colab/RUN_V2_INSTRUCTIONS.md         how to run it in Google Colab
colab/ea_gatsched_v3_residual.py     optional extension: residual-over-SJBF policy (see below; not part of the reported results)
colab/README_V3_RESIDUAL.md          design and usage rules for the optional extension
results_v2/                          outputs of the reported run (per-window results, summaries, run log)
archive_v1_first_run/                outputs of the first run (20 Polaris test windows), kept for the sensitivity check
analysis/analyze.py                  pooled ratios, bootstrap CIs, Wilcoxon with Holm correction, strata, seeds
analysis/other_metrics.py            95th-percentile wait, utilization, throughput, makespan
analysis/make_sens_table.py          sensitivity table (first run versus reported run)
analysis/make_figures.py             figures in figs/
analysis/make_tables.py              LaTeX tables in tables/
analysis/geopm_validate_full.py      GEOPM power-level validation (Kaggle notebook code)
analysis/geopm_*.json, *.csv         outputs of the GEOPM validation
analysis/*.csv                       pooled, per-seed, strata, energy-sensitivity and agreement tables
figs/, tables/                       generated figures and tables used by the manuscript
```

## Reproducing the reported results

1. Obtain the ALCF job logs (`DIM_JOB_COMPOSITE`: Polaris, Mira, Cooley) from the ALCF Data Catalog on IEEE DataPort (DOI 10.21227/bhfr-wx19). They are not redistributed here.
2. Run `colab/ea_gatsched_v2.py` in Google Colab, following `colab/RUN_V2_INSTRUCTIONS.md`. Put the four `ANL-ALCF-DJC-*.csv.gz` files in `/content/data` or set `EA_DATA_DIR`. Use `EA_FAST=1` for a short smoke test. The reported run took 10,043 s on a Colab runtime. Save the results zip to Drive as soon as the run finishes.
3. Copy the output CSVs into `results_v2/`, then from the repository root run:

```
python analysis/analyze.py
python analysis/other_metrics.py
python analysis/make_sens_table.py
python analysis/make_tables.py
python analysis/make_figures.py
```

Key settings: windows of 1,500 jobs; chronological 60/10/30 split into train, validation and test; all non-overlapping test windows (Polaris 45, Mira 10, Cooley 19); 120 training episodes; seeds 0, 1, 2; 85% power cap; P_busy = 149.48 W and P_idle = 69.86 W (GEOPM Theta telemetry, applied to all machines as an assumption).

Notes:
- `results_v2/paired_vs_easy.csv` (from the Colab run) averages per-window relative changes. The paper uses pooled ratios of means from `analysis/pairwise_pooled.csv`.
- The `CE nan` lines in the training log come from a window with no multi-candidate decisions. They are not a training failure.
- Statistics: pooled ratio of means, paired bootstrap (10,000 resamples of windows), two-sided Wilcoxon signed-rank tests, and Holm correction within pre-specified families.

## GEOPM power-level validation (Kaggle)

The simulator's node-power levels come from Theta GEOPM power-balancer traces (DGEMM, 155 W balancer setting, 8,640 trace files, 4,320 nodes, 53.5 million rows).

- Kaggle notebook: https://www.kaggle.com/code/kyrianadimora/geopm-validation
- Kaggle dataset: https://www.kaggle.com/datasets/kyrianadimora/alcf-geopm-power-balancer-155w
- Script (same code as the notebook): `analysis/geopm_validate_full.py`
- Outputs stored here: `analysis/geopm_validation.json`, `analysis/geopm_data_quality.csv`, `analysis/geopm_per_node.csv`. The notebook also writes a per-file CSV, a feature-correlation CSV and a LaTeX table, which are not stored here.
- Reported in Appendix B of the paper. The run takes about 100 minutes on Kaggle for the complete dataset.

Main results: mean package power 149.46 W (standard deviation 15.45 W, minimum 44.81 W, maximum 177.15 W); per-node mean standard deviation 5.70 W; 38.5% of samples above 155 W, because the setting is an average budget that GEOPM redistributes among nodes; reported power agrees with the power implied by the energy counter with a mean absolute error of 6.15 W (4.19%). The dataset has one benchmark at one cap and no idle periods, so it cannot validate the idle level or a sweep over caps.

## Optional extension: residual-over-SJBF policy (GAT-BF-R)

`colab/ea_gatsched_v3_residual.py` trains a policy whose starting point is exactly SJBF (logit = minus the SJBF rank plus a learned residual initialised at zero). REINFORCE fine-tunes the residual against SJBF, and a fine-tuned checkpoint replaces SJBF only if it beats SJBF on the validation windows by a fixed margin. Windows, metrics and statistics are unchanged. The design constants are fixed in advance and the script is meant to be run once and reported whatever the outcome; see `colab/README_V3_RESIDUAL.md`. Results from this script are not part of the reported results above.

## Scope and limitations

- The simulator omits priorities, fair-share, reservations, maintenance, node failures and topology.
- The power cap only reduces capacity (it never slows jobs). The cap bounds mean power, not instantaneous power.
- Node power is constant per state and taken from a different machine (Theta KNL, DGEMM, one cap). Idle power is an assumption, and the energy sweep varies the idle-to-busy ratio from 0.2 to 0.8.
- GAT-BF orders backfill candidates only; it does not choose nodes, frequencies or the head job.
- Mira has 10 test windows and Cooley 19, so intervals there are wide.
- Per-decision scheduling time was not measured.

## Data and licence notes

The ALCF job logs and the GEOPM traces are not redistributed here. See the ALCF Data Catalog and the GEOPM dataset pages for their terms.

## Acknowledgments

This research is supported in part by the US National Science Foundation under Grant #2441633. The authors acknowledge the computational resources and workload traces provided by the Argonne Leadership Computing Facility.
