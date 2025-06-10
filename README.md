# EA-GATSched: Energy-Aware Adaptive Scheduler Implementation
A novel HPC scheduling framework leveraging Graph Attention Networks (GAT) and Multi-Objective Reinforcement Learning (MORL) to intelligently optimize both energy consumption and computational performance.

# https://www.python.org/downloads/
# https://pytorch.org/

# Overview
EA-GATSched introduces an energy-aware Graph Attention Network scheduler that revolutionizes job scheduling in High-Performance Computing (HPC) systems. Our approach achieves 28-35% energy consumption reduction compared to baseline SLURM-like schedulers while improving traditional performance metrics.

# Key Features
a)  Graph Attention Networks: Intelligent job scheduling using GAT-based decision making
b)  Energy Optimization: Multi-objective policy network balancing performance and energy efficiency
c)  System-Specific Tuning: Optimized configurations for different HPC architectures
d)  Comprehensive Benchmarking: Quantitative evaluation against traditional scheduling systems and ML-based Scheduling Approaches
e)  Easy Deployment: Ready-to-use Google Colab implementation

# Main Contributions
C₁: Energy-aware GAT scheduler optimizing job scheduling with energy efficiency, performance, and load balancing
C₂: Multi-objective policy network adapting decisions based on system configurations and runtime conditions
C₃: Machine-specific parameter configurations for Polaris, Mira, and Cooley supercomputing systems
C₄: Improved energy consumption modeling preventing unrealistic estimates
C₅: Comprehensive benchmarking framework demonstrating significant improvements over SLURM-like systems and ML-based Scheduling Approaches

# Quick Start
Option 1: Google Colab (Recommended)
The fastest way to get started:
1. Open our Colab notebook:
   # https://colab.research.google.com/drive/1xmEG9gnKiNIKOSs0DHWnfO654NC7awG2#scrollTo=UmagVTKMNuzb
2. Select GPU runtime: Runtime → Change runtime type → GPU
3. Run all cells: Runtime → Run all
4. Follow the notebook: All datasets, training, and visualizations are included

# Option 2: Local Installation
# Prerequisites
# Hardware Requirements:

a) CUDA-capable GPU with 8GB+ memory (recommended)
b) 16GB RAM minimum (32GB recommended)
c) 4+ core CPU (8+ cores recommended)

# Software Requirements:
a) Python 3.8+
b) CUDA Toolkit 11.8+

# Installation Steps

# Clone the repository
git clone https://github.com/yourusername/EA-GATSched-Energy-Aware-Adaptive-Scheduler-Implementation.git
cd EA-GATSched-Energy-Aware-Adaptive-Scheduler-Implementation

# Create virtual environment
python -m venv ea_gat_env
source ea_gat_env/bin/activate  # On Windows: ea_gat_env\Scripts\activate

# Install dependencies
pip install torch==2.1.0 torch-geometric==2.4.0 pandas==2.1.1 numpy==1.24.3 \
    matplotlib==3.8.0 seaborn==0.12.2 scikit-learn==1.3.0 tqdm==4.66.1

# Download Datasets
# Create data directory
mkdir -p data

# Download HPC workload traces
wget -O data/HPC-POLARIS_traces.csv https://reports.alcf.anl.gov/data/HPC-POLARIS_traces.csv
wget -O data/HPC-MIRA_traces.csv https://reports.alcf.anl.gov/data/HPC-MIRA_traces.csv
wget -O data/HPC-COOLEY_traces.csv https://reports.alcf.anl.gov/data/HPC-COOLEY_traces.csv

# Usage
Basic Usage
pythonfrom src.energy_aware_scheduler import EnergyAwareScheduler

# Initialize scheduler
scheduler = EnergyAwareScheduler([
    'data/HPC-POLARIS_traces.csv',
    'data/HPC-MIRA_traces.csv',
    'data/HPC-COOLEY_traces.csv'
])

# Load and preprocess data
scheduler.load_and_preprocess_data()

# Train models for each system
for machine_name in ['POLARIS', 'MIRA', 'COOLEY']:
    dataset = scheduler.datasets[f'data/HPC-{machine_name}_traces.csv']
    scheduler.train_model(machine_name, dataset)

# Evaluate performance
results = {}
for machine_name in ['POLARIS', 'MIRA', 'COOLEY']:
    results[machine_name] = scheduler.evaluate_model(machine_name)

# Generate visualizations
scheduler.plot_energy_efficiency()
scheduler.plot_throughput_comparison()
scheduler.plot_resource_utilization()

# Expected Runtime
# Task                           Time Estimate 
Dataset Setup:                   15-30 minutes 
Model Training:                  3-12 hours
Evaluation & Analysis:           1-4 hours
Training time varies by dataset size and hardware

# Architecture
Workflow Overview
Data Preprocessing → Graph Construction → Model Training → Performance Evaluation
       ↓                    ↓               ↓                    ↓
   Feature Engineering   Node/Edge      GAT Training      Visualization
   Energy Calculation   Creation       Policy Optimization   Benchmarking
   
Key Components
1. EnergyAwareGATScheduler: Core GAT-based scheduling algorithm
2. MultiObjectivePolicyNetwork: Balances energy and performance objectives
3. SystemSpecificConfigurator: Applies machine-specific optimizations
4. EnergyConsumptionModeler: Accurate energy estimation and modeling
5. BenchmarkingFramework: Comprehensive performance evaluation

# Outputs
The system generates:

# Performance Metrics
a) Energy consumption reduction percentages
b) Job throughput measurements
c) Resource utilization statistics
d) Load balancing scores
e) Waiting time improvements

# Visualizations
a) Energy efficiency over time plots
b) Resource utilization heat maps
c) Multi-objective optimization trade-offs
d) Comparative analysis charts

# Model Artifacts
a) Trained PyTorch models for each HPC system
b) Configuration files with optimized parameters
c) Evaluation results in CSV format

# Validation
To validate results against published research:
1. Energy Reduction: Compare with Table 4 values (28-35% improvement)
2. System Configurations: Verify optimal results from machine-specific tuning
3. Energy Modeling: Confirm realistic energy estimates without outliers
4. Multi-objective Policy: Validate adaptive decision-making capabilities
5. Visualizations: Match patterns in Figures 4-7 from the paper

# Contributing
We welcome contributions! Please see our Contributing Guidelines for details.
How to Contribute
Types of Contributions
We welcome several types of contributions:
a) Bug fixes: Fix identified issues in the codebase
b) Feature enhancements: Add new scheduling algorithms or optimization techniques
c) Performance improvements: Optimize energy consumption or execution time
d) Documentation: Improve README, docstrings, or add tutorials
e) Tests: Add or improve test coverage
Examples: Provide usage examples or case studies

# Development Setup

# Clone and setup development environment
git clone https://github.com/yourusername/EA-GATSched-Energy-Aware-Adaptive-Scheduler-Implementation.git
cd EA-GATSched-Energy-Aware-Adaptive-Scheduler-Implementation

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black src/
flake8 src/

# Acknowledgments
Argonne Leadership Computing Facility for providing HPC workload datasets and University of Kansas for providing platform and support.

