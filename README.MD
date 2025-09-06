# Incremental PINNs for Population Balance Models

This project implements an **Incremental Physics-Informed Neural Network (iPINN)** to solve Population Balance Models (PBMs). The framework is designed to simulate complex particle processes like **aggregation** and **breakage**. A key feature is its ability to learn incrementally, solving multiple tasks with different physical parameters or initial conditions sequentially, using network pruning and masking to retain knowledge from previous tasks.

## Key Features

- **Physics-Informed Neural Networks (PINNs):** Solves differential equations by incorporating them into the neural network's loss function.
- **Population Balance Models (PBMs):** Accurately models systems of particles undergoing aggregation and breakage.
- **Incremental Learning:** Capable of learning a sequence of tasks by freezing important weights from previous tasks and growing the network for new ones.
- **Network Pruning:** Implements an importance-based pruning algorithm to identify and preserve the most critical neurons for a given task.
- **Customizable Simulations:** Easily configure PBM parameters, initial conditions, and neural network architecture through command-line arguments.
- **Visualization:** Automatically generates and saves plots, including heatmaps of the solution, time-slice comparisons, and loss histories.

---

## Installation

Follow these steps to set up the project environment. A virtual environment is highly recommended.

1.  **Clone the repository:**
    ```powershell
    git clone https://github.com/dimasavr2006/iPINN-PBMs.git
    cd iPINN-PBMs
    ```

2.  **Create and activate a virtual environment:**
    *   **Windows:**
        ```powershell
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        ```
    *   **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required packages:**
    The project uses PyTorch, Torch-Optimizer, and other scientific computing libraries.
    ```powershell
    pip install -r requirements.txt
    ```

---

## Usage

All experiments are run from the `src/` directory using `main.py`. You can specify the task type, model architecture, and other hyperparameters via command-line arguments.

### Example 1: Pure Aggregation Task

This command trains a model to solve a PBM dominated by the aggregation process, using the initial condition proposed by Chen et al.

```powershell
python src/main.py --task_type aggregation `
                   --ic_type chen_agg `
                   --num_tasks 1 `
                   --ic_mus 1.0 `
                   --ic_sigmas 0.05 `
                   --num_epochs_train 3000 `
                   --num_epochs_retrain 500 `
                   --optimizer_name Adam `
                   --lr 0.001 `
                   --a 0.0 `
                   --b 1.0 `
                   --v_min 0.001 `
                   --v_max 10.0 `
                   --xgrid 256 `
                   --nt 101 `
                   --N_f 4096 `
                   --layers 40,40,40,40,40,40 `
                   --num_v_quad 256 `
                   --seed 996 `
                   --lbfgs_max_iter 1000
```

### Example 2: Pure Breakage Task

This command trains a model to solve a PBM dominated by the particle breakage process, starting from a Dirac delta-like initial condition.

```powershell
python src/main.py --task_type breakage `
                   --ic_type chen_break `
                   --num_tasks 1 `
                   --ic_mus 1.0 `
                   --ic_sigmas 0.05 `
                   --num_epochs_train 3000 `
                   --num_epochs_retrain 500 `
                   --optimizer_name Adam `
                   --lr 0.001 `
                   --a 0.0 `
                   --b 1.0 `
                   --v_min 0.001 `
                   --v_max 1.0 `
                   --xgrid 256 `
                   --nt 101 `
                   --N_f 4096 `
                   --layers 40,40,40,40,40,40 `
                   --num_v_quad 256 `
                   --seed 996 `
                   --lbfgs_max_iter 1000
```

---

## Command-Line Arguments

The behavior of the simulation can be controlled with the following arguments:

| Argument               | Description                                                                 | Default Value            |
| ---------------------- | --------------------------------------------------------------------------- | ------------------------ |
| `--seed`               | Random seed for reproducibility.                                            | `42`                     |
| `--num_tasks`          | Number of incremental learning tasks to perform.                            | `1`                      |
| `--task_type`          | Type of PBM: `aggregation`, `breakage`, or `combined`.                      | `aggregation`            |
| `--ic_type`            | Initial condition: `gaussian`, `chen_agg`, `chen_break`.                      | `gaussian`               |
| `--ic_mus`             | Comma-separated list of means (μ) for the initial conditions of each task.  | `"2.0"`                  |
| `--ic_sigmas`          | Comma-separated list of stds (σ) for the initial conditions of each task.   | `"0.5"`                  |
| `--lr`                 | Learning rate for the Adam optimizer.                                       | `1e-3`                   |
| `--num_epochs_train`   | Number of training epochs for the initial stage.                            | `3000`                   |
| `--num_epochs_retrain` | Number of retraining epochs after pruning (for multi-task learning).        | `2000`                   |
| `--lbfgs_max_iter`     | Maximum iterations for the L-BFGS optimizer fine-tuning step.               | `5000`                   |
| `--layers`             | Comma-separated string defining the hidden layer sizes (e.g., `40,40,40`).  | `"40,40,40,40,40,40,40,40"` |
| `--optimizer_name`     | Choice of optimizer (e.g., `Adam`, `LBFGS`).                                | `Adam`                   |
| `--weight_decay`       | L2 regularization factor.                                                   | `0.0`                    |
| `--alpha_fc`           | Pruning threshold parameter (retains connections accounting for `alpha_fc` of importance). | `0.95`            |
| `--v_min`, `--v_max`   | Min and max particle volume for the simulation domain.                        | `0.001`, `10.0`          |
| `--a`, `--b`           | Start and end time for the simulation domain.                               | `0.0`, `1.0`             |
| `--N_f`                | Number of collocation points for training (physics loss).                   | `4096`                   |
| `--num_v_quad`         | Number of grid points for numerical integration in the loss function.       | `512`                    |
| `--visualize`          | Set to `True` to generate and save result plots.                            | `True`                   |


---

## Expected Output

When the script is run with `--visualize True`, it will generate and save several plots in a directory named `pbm_results/`. The results are organized by experiment name and task number.

The output directory will look like this:
```
pbm_results/
└── PBM_tanh_alpha0.95_wd0.0_1tasks_seed996/
    └── after_task_1/
        ├── heatmaps_combined_task1.pdf
        ├── slices_subplots_task1.pdf
        └── loss_history.pdf
```

- **`heatmaps_combined_taskX.pdf`**: Shows the exact solution, the PINN prediction, and the absolute error as heatmaps.
- **`slices_subplots_taskX.pdf`**: Compares the exact vs. predicted solution at different time slices.
- **`loss_history.pdf`**: Plots the total, initial condition, and physics residual loss over training epochs.

Additionally, model checkpoints (`.pth`) and mask files (`.pt`) are saved in the project's root directory.

---

## Project Structure

```
├── src/
│   ├── main.py                   # Main script to run experiments
│   ├── ipinn.py                  # iPINN model and DNN architecture with masking
│   ├── incremental_learning.py   # Training loop for sequential tasks
│   ├── pbm_solver.py             # PBM kernels and numerical solver for ground truth data
│   ├── pruning.py                # Neuron and weight pruning logic
│   ├── choose_optimizer.py       # Helper for selecting optimizers
│   ├── visualize.py              # Plotting and visualization functions
│   └── utils.py                  # Utility functions (seeding, error metrics, etc.)
│
├── requirements.txt              # Python package dependencies
└── README.md                     # This file
```
