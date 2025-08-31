import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_pbm_combined_heatmaps(N_exact, N_pred, N_diff, task_id, v_grid, t_grid, pbm_params, path, save=False):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    data_list = [N_exact, N_pred, N_diff]
    title_list = ['Exact Solution', 'PINN Prediction', 'Absolute Error']
    cmap_list = ['rainbow', 'rainbow', 'magma']

    for i, ax in enumerate(axes):
        data = data_list[i]
        title = title_list[i]
        cmap = cmap_list[i]
        h = ax.imshow(data.T, interpolation='nearest', cmap=cmap,
                      extent=[t_grid.min(), t_grid.max(), v_grid.min(), v_grid.max()],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=14)
        ax.set_xlabel('Time (t)', fontweight='bold', size=16)
        ax.set_title(title, fontsize=18)
        ax.tick_params(labelsize=14)

    axes[0].set_ylabel('Particle Volume (v)', fontweight='bold', size=16)

    ic_mu = pbm_params.get('ic_mu', 'N/A')
    ic_sigma = pbm_params.get('ic_sigma', 'N/A')

    fig.suptitle(f"Task {task_id + 1}: IC μ={ic_mu}, σ={ic_sigma}", fontsize=22, y=1.02)
    plt.tight_layout()

    if save:
        filename = f"heatmaps_combined_task{task_id + 1}.pdf"
        plt.savefig(f"{path}/{filename}")
        print(f"Saved combined heatmap to {path}/{filename}")

    plt.show()
    plt.close()


def plot_pbm_time_slices_subplots(N_data_dict, task_id, v_grid, t_grid, pbm_params, path, save=False):

    v0 = pbm_params.get('ic_mu')
    v_grid_normalised = v_grid / v0

    num_t = len(t_grid)
    time_indices = [0, num_t // 3, 2 * num_t // 3, num_t - 1]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)

    for i, t_idx in enumerate(time_indices):
        ax = axes[i]
        time_val = t_grid[t_idx]


        ax.plot(v_grid_normalised, N_data_dict['exact'][t_idx, :], 'b-', linewidth=2, label='Exact solution')
        ax.plot(v_grid_normalised, N_data_dict['predicted'][t_idx, :], 'r--', linewidth=2, label='PINN prediction')
        ax.set_xlabel('Normalised volume (v/v0) should be')


        # ax.plot(v_grid, N_data_dict['exact'][t_idx, :], 'b-', linewidth=2, label='Exact')
        # ax.plot(v_grid, N_data_dict['predicted'][t_idx, :], 'r--', linewidth=2, label='Predicted')
        # ax.set_xlabel('Particle Volume (v)', fontweight='bold', size=16)


        ax.set_title(f'Time t = {time_val:.2f}', fontsize=18)
        ax.tick_params(labelsize=14)
        ax.legend()
        ax.grid(True, linestyle=':')

    axes[0].set_ylabel('Number Density N(v,t)', fontweight='bold', size=16)

    ic_mu = pbm_params.get('ic_mu', 'N/A')
    ic_sigma = pbm_params.get('ic_sigma', 'N/A')
    fig.suptitle(f"Time Slices Comparison\nTask {task_id + 1}: IC μ={ic_mu}, σ={ic_sigma}", fontsize=22, y=1.04)
    plt.tight_layout()

    if save:
        filename = f"slices_subplots_task{task_id + 1}.pdf"
        plt.savefig(f"{path}/{filename}")
        print(f"Saved time slices subplots to {path}/{filename}")

    plt.show()
    plt.close()


def plot_loss_history(loss_histories, path, experiment_name, save=False):
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    loss_types = ['total', 'ic', 'f']
    loss_titles = ['Total Loss', 'Initial Condition Loss (IC)', 'Physics Residual Loss (f)']
    num_tasks = len(loss_histories)

    for i, loss_type in enumerate(loss_types):
        ax = axes[i]
        for task_id, history in loss_histories.items():
            steps = np.arange(len(history[loss_type])) * 100
            ax.plot(steps, history[loss_type], label=f'Task {task_id + 1}')

        ax.set_ylabel('Loss Value', fontweight='bold', size=16)
        ax.set_title(loss_titles[i], fontsize=20)
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--")
        ax.tick_params(labelsize=14)
        if num_tasks > 1:
            ax.legend(fontsize=12)

    axes[-1].set_xlabel('Epoch', fontweight='bold', size=18)
    fig.suptitle('Training Loss Components vs. Epochs', fontsize=24, y=1.01)
    plt.tight_layout()

    if save:
        filename = "loss_history.pdf"
        plt.savefig(f"{path}/{filename}")
        print(f"Saved detailed loss history plot to {path}/{filename}")

    plt.show()
    plt.close()


def visualize_pbm_results(model, args, pbm_tasks_params, VT_star, N_star, loss_histories, current_task_id):
    base_path = f"pbm_results/{model.experiment_name}"
    path = f"{base_path}/after_task_{current_task_id + 1}"
    if not os.path.exists(path):
        os.makedirs(path)


    for task_id in range(current_task_id + 1):
        print(f"    - Visualizing performance on Task {task_id + 1}")
        model.set_task(task_id)

        vt_task = VT_star[task_id]
        n_exact_flat = N_star[task_id]
        n_pred_flat = model.predict(vt_task)

        num_v = args.xgrid
        num_t = args.nt

        v_grid = vt_task[:num_v, 0]
        t_grid = np.unique(vt_task[:, 1])

        N_exact_grid = n_exact_flat.reshape(num_t, num_v)
        N_pred_grid = n_pred_flat.reshape(num_t, num_v)
        N_diff_grid = np.abs(N_exact_grid - N_pred_grid)

        current_params = pbm_tasks_params[task_id]

        plot_pbm_combined_heatmaps(N_exact_grid, N_pred_grid, N_diff_grid,
                                   task_id, v_grid, t_grid, current_params, path, save=True)

        data_for_slices = {'exact': N_exact_grid, 'predicted': N_pred_grid}
        plot_pbm_time_slices_subplots(data_for_slices, task_id, v_grid, t_grid, current_params, path, save=True)

    plot_loss_history(loss_histories, path, model.experiment_name, save=True)