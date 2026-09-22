# EA-GATSched: Energy-Aware Adaptive Scheduler Implementation

A graph-attention (GATv2) and adaptive multi-objective reinforcement-learning
(PPO) scheduler for energy-aware HPC batch scheduling. This repository
accompanies the manuscript *"EA-GATSched: Graph Attention Networks and
Adaptive Multi-Objective Reinforcement Learning for Energy-Aware HPC Batch
Scheduling"* (Kyrian C. Adimora and Hongyang Sun, University of Kansas)*.

- **Implementation notebook**: [`Energy_Aware_Adaptive_Scheduler_Implementation.ipynb`](./Energy_Aware_Adaptive_Scheduler_Implementation.ipynb)
- **GEOPM power-model fitting notebook (Kaggle)**: https://www.kaggle.com/code/kyrianadimora/notebook4053448c85/edit

## Overview

EA-GATSched combines three pieces to lower facility energy under a hard
power budget:

1. A **GATv2 graph encoder** that represents queued and running jobs
   together with their resource conflicts and dependencies as a dynamic
   job graph (Section 4 of the paper).
2. A **component-level power model** whose functional form and free
   parameters are fit to real GEOPM node telemetry and validated on a
   disjoint, node-grouped hold-out set (Section 5).
3. A **PPO policy** whose three reward terms -- energy, throughput,
   utilization -- are combined with weights produced by a small
   state-conditioned network, rather than fixed in advance (Section 7).

Replaying 389,604 production jobs from three ALCF systems (Polaris, Mira,
Cooley) in a discrete-event simulator, EA-GATSched reduces energy by
**27.3-56.6%** against four classical schedulers (SLURM, PBS Pro, LSF,
Volcano) and by **14.3-30.2%** against two learned baselines (DeepRM,
Decima), at a throughput cost on two of the three systems (Table 6 of the
paper). A reduced-scale, fully executed diagnostic study (Section 9)
additionally shows where the adaptive weighting sits relative to a swept
fixed-weight Pareto front, and how scheduling quality degrades as
power-model error grows.

## Key Contributions

- **C1**: A GATv2-based dynamic job graph encoder over queued/running jobs,
  their resource conflicts, and workflow dependencies.
- **C2**: A component-level GEOPM-calibrated power model, validated with a
  real, node-grouped hold-out (not just fit residuals).
- **C3**: A state-conditioned, adaptive objective-weighting scheme for the
  PPO reward, evaluated against a swept fixed-weight Pareto front.
- **C4**: A 389,604-job, three-system (Polaris/Mira/Cooley) evaluation
  against six re-implemented baselines (SLURM, PBS Pro, LSF, Volcano,
  DeepRM, Decima).
- **C5**: A reduced-scale but fully executed diagnostic simulator releasing
  runnable code for the Pareto sweep, power-model error-injection
  sensitivity, and weight-dynamics analyses.

## Repository Contents

This repository currently holds the implementation notebook and its
documentation:

```
.
├── Energy_Aware_Adaptive_Scheduler_Implementation.ipynb   # Full implementation (data prep, GATv2 encoder,
│                                                           #   PPO training, evaluation, plotting)
└── README.md
```

Key components defined in the notebook: `EnergyAwareGATEncoder`,
`EnergyAwareGATScheduler`, `MultiObjectivePolicyNetwork`, `PowerProfile`,
`EnergyConfig`, `MultiObjectiveMetrics`, `JobSchedulerDataset` /
`EnhancedJobSchedulerDataset`, and `SchedulingSystem` /
`EnhancedSchedulingSystem`.

The manuscript's figure-generation scripts (`make_power_kaggle_figure.py`,
`make_figures.py`, `reference_simulator.py`) and the LaTeX source live in
the paper's companion project and are not part of this notebook repo; if
you would like them added here as well (for full end-to-end
reproducibility of every manuscript figure from one place), open an issue
and they can be merged in.

## Quick Start

### Option 1: Google Colab / Jupyter (recommended)

1. Open `Energy_Aware_Adaptive_Scheduler_Implementation.ipynb` directly in
   Google Colab (via GitHub → Open in Colab) or in a local Jupyter
   environment.
2. Select a GPU runtime: Runtime -> Change runtime type -> GPU.
3. Run all cells. The notebook loads and preprocesses the workload traces,
   builds the job graph, trains the GATv2 + PPO scheduler, and produces the
   evaluation plots described below.

### Option 2: Local installation

**Hardware.** A CUDA-capable GPU with 8GB+ memory is recommended for
training; the notebook also runs on CPU (slower). 16GB+ RAM recommended.

**Software.** Python 3.8+, and the packages the notebook imports:

```bash
git clone https://github.com/myandelaepu/EA-GATSched-Energy-Aware-Adaptive-Scheduler-Implementation.git
cd EA-GATSched-Energy-Aware-Adaptive-Scheduler-Implementation

python -m venv ea_gat_env
source ea_gat_env/bin/activate  # On Windows: ea_gat_env\Scripts\activate

pip install torch torch-geometric pandas numpy matplotlib seaborn \
    scikit-learn networkx tqdm jupyter
```

Then launch Jupyter and run the notebook top to bottom:

```bash
jupyter notebook Energy_Aware_Adaptive_Scheduler_Implementation.ipynb
```

### Workload traces

The manuscript's main results (Table 6) use production job traces from
three ALCF systems, summarized in Table 5 of the paper:

| System  | Jobs    | Period    | Nodes  | Hardware        |
|---------|---------|-----------|--------|-----------------|
| Polaris | 241,772 | 2024      | 560    | A100, HDR IB    |
| Mira    | 52,154  | 2018-19   | 49,152 | Blue Gene/Q     |
| Cooley  | 95,678  | 2018-19   | 126    | Haswell         |

These are ALCF operational job traces and are not redistributed in this
repository. They are available from the ALCF reporting dashboards
(https://reports.alcf.anl.gov), which are interactive HTML/BI report pages
rather than direct file endpoints -- so obtaining an extract means using
the dashboard's own export/query tools for the relevant system and date
range, not a single `wget`/`curl` command against the dashboard URL.

## GEOPM Power-Model Fitting

The component-level power model used throughout the paper,

```
P(f) = 149.481 + 9.692e-22 * f^1.199   [W], f in Hz
```

was fit to 8,555,779 raw GEOPM samples (3,456 nodes) from the public ALCF
GEOPM dataset (`power_balancer` policy, 155 W cap, `dgemm` benchmark) and
validated on a disjoint, **node-grouped** hold-out of 2,139,222 samples (864
nodes): MAE = 8.93 W, RMSE = 15.42 W, MAPE = 7.33% (Table 2 and Figure 1 of
the paper). The data-parsing and curve-fitting pipeline is a runnable
Kaggle notebook:

https://www.kaggle.com/code/kyrianadimora/notebook4053448c85/edit

## Validating Results Against the Paper

When checking notebook output against the manuscript, use these figure/table
references (not the numbers from earlier drafts of this README):

1. **Power-model accuracy**: Table 2 and Figure 1 -- MAE 8.93 W, RMSE
   15.42 W, MAPE 7.33% on the real, node-grouped hold-out (not 28-35%,
   which was an earlier placeholder figure for a different metric).
2. **Energy reduction**: Table 6 -- 27.3-56.6% vs. the four classical
   baselines, 14.3-30.2% vs. the two learned baselines, per system.
3. **Ablation**: Table 7 and Figure 8 -- component-wise contribution to
   the energy reduction on Polaris.
4. **Scalability**: Table 8 and Figure 9 -- GATv2 inference latency vs.
   node count.
5. **Diagnostic study**: Section 9 and Figures 10-12 -- Pareto
   positioning, power-model error-injection sensitivity, and weight
   dynamics, on the reduced-scale reference simulation (explicitly not a
   re-run of the full 389,604-job ALCF study; see Section 10 for the
   itemized simulator scope).

## Outputs

Running the notebook end to end produces:

- Trained GATv2 + PPO model checkpoints per system (Polaris, Mira, Cooley).
- Evaluation metrics: energy consumption, throughput, utilization, and
  mean wait time, relative to each baseline.
- The plots underlying Figures 6-9 of the paper (energy savings,
  energy/throughput trade-off, ablation, scalability).

## Contributing

Bug reports and pull requests are welcome -- in particular, contributions
that factor the notebook into a proper `src/` package with tests would be
very useful for reuse. Please open an issue describing the change before
submitting a substantial PR.

## Citation

If you use this work, please cite:

```bibtex
@article{Adimora2027EAGATSched,
  author  = {Kyrian C. Adimora and Hongyang Sun},
  title   = {{EA-GATSched}: Graph Attention Networks and Adaptive
             Multi-Objective Reinforcement Learning for Energy-Aware
             {HPC} Batch Scheduling},
  journal = {Future Generation Computer Systems},
  year    = {2027}
}
```

(Volume/pages/DOI will be added once assigned by the journal.)

## Acknowledgments

Argonne Leadership Computing Facility for the Polaris, Mira, and Cooley
operational job traces and the public GEOPM telemetry dataset, and the
University of Kansas Department of Electrical Engineering and Computer
Science for platform and support.
