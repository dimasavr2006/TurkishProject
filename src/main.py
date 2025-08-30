import argparse
import numpy as np
import os
import torch
from utils import *
from incremental_learning import incremental_learning
from ipinn import iPINN
from visualize import visualize_pbm_results
from pbm_solver import *
from scipy.stats import qmc

torch.backends.cudnn.benchmark = True

def process_data(args, pbm_tasks_params, initial_condition_func):
    VT_star_list, N_star_list = [], [] # эталонные списки для данных для разного числа задач
    VT_ic_train_list, N_ic_train_list = [], []
    VT_f_train_list = []

    v_domain = (args.v_min, args.v_max) # границы расчётной области
    t_domain = (args.a, args.b)

    print("--- Generating data for PBM tasks ---")
    for task_id, params in enumerate(pbm_tasks_params):
        print(f"Generating data for Task {task_id + 1} with params: {params}")

        v_grid_exact, t_grid_exact, N_exact = generate_pbm_solution(
            pbm_params=params,
            v_domain=v_domain, num_v=args.xgrid,
            t_domain=t_domain, num_t=args.nt,
            initial_condition_func=initial_condition_func,
            device='cpu'
        ) # результат: вектор, вектор и матрица

        V_star, T_star = np.meshgrid(v_grid_exact.numpy(), t_grid_exact.numpy()) # из двух векторов делаемм сетку
        VT_star = np.hstack((V_star.flatten()[:, None], T_star.flatten()[:, None]))
        N_star = N_exact.numpy().flatten()[:, None]
        VT_star_list.append(VT_star) # готовые данные для выбранной задачи добавляются в списки
        N_star_list.append(N_star)

        v_ic = v_grid_exact.numpy()
        t_ic = np.zeros_like(v_ic) # начальные условия

        VT_ic_train = np.hstack((v_ic[:, None], t_ic[:, None])) # массив координат
        N_ic_train = N_exact.numpy()[0, :].flatten()[:, None]
        VT_ic_train_list.append(VT_ic_train) # добавление массивов в списки
        N_ic_train_list.append(N_ic_train)

        sampler = qmc.LatinHypercube(d=2, seed=args.seed) # создаём коллокационные точки
        sample = sampler.random(n=args.N_f)

        v_f_train = sample[:, 0] * (args.v_max - args.v_min) + args.v_min
        t_f_train = sample[:, 1] * (args.b - args.a) + args.a # тут масштабируют данные

        t_f_train[0] = args.a # первая точка по времени принудительно устанавливается

        VT_f_train = np.hstack((v_f_train[:, None], t_f_train[:, None]))
        VT_f_train_list.append(VT_f_train) # массив добавляется в список, ля коллокационных точек нам не нужны значения N, так как именно в этих точках мы будем вычислять невязку residual, которая и есть наша цель для минимизации

    return VT_star_list, N_star_list, VT_ic_train_list, N_ic_train_list, VT_f_train_list


def main():
    parser = argparse.ArgumentParser(description='Incremental PINNs for PBM')

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--num_tasks', type=int, default=1, help='Number of tasks with different ICs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')

    parser.add_argument('--num_epochs_train', type=int, default=3000, help='Number of epochs for initial training.')
    parser.add_argument('--num_epochs_retrain', type=int, default=2000,
                        help='Number of epochs for retraining after pruning.')

    parser.add_argument('--layers', type=str, default='40,40,40,40,40,40,40,40',
                        help='Comma-separated list of neuron counts for hidden layers.')

    parser.add_argument('--ic_mus', type=str, default="2.0", help='Comma-separated means for initial conditions.')
    parser.add_argument('--ic_sigmas', type=str, default="0.5", help='Comma-separated stds for initial conditions.')

    parser.add_argument('--a', type=float, default=0.0, help='Start of time interval.')
    parser.add_argument('--b', type=float, default=1.0, help='End of time interval.')
    parser.add_argument('--v_min', type=float, default=0.001, help='Min particle volume.')
    parser.add_argument('--v_max', type=float, default=10.0, help='Max particle volume.')
    parser.add_argument('--xgrid', type=int, default=256, help='Grid points for data generation.')
    parser.add_argument('--nt', type=int, default=101, help='Time points for data generation.')
    parser.add_argument('--N_f', type=int, default=4096, help='Number of collocation points.')
    parser.add_argument('--num_v_quad', type=int, default=512, help='Grid points for integration in loss.')

    parser.add_argument('--optimizer_name', type=str, default='Adam', help='Optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--alpha_fc', type=float, default=0.95, help='Pruning parameter.')
    parser.add_argument('--net', type=str, default='DNN', help='Net architecture.')
    parser.add_argument('--activation', default='tanh', help='Activation function.')
    parser.add_argument('--loss_style', default='mean', help='Loss aggregation style.')
    parser.add_argument('--visualize', default=True, help='Visualize the solution.')

    parser.add_argument('--lbfgs_max_iter', type=int, default=5000, help='Max iterations for L-BFGS optimizer.')

    parser.add_argument('--task_type', type=str, default='aggregation', help='Type of PBM task: "aggregation", "breakage", or "combined".')
    parser.add_argument('--ic_type', type=str, default='gaussian', help='Initial condition type: "gaussian", "chen_agg", "chen_break".')

    args = parser.parse_args()


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    mus = [float(mu) for mu in args.ic_mus.split(',')]
    sigmas = [float(s) for s in args.ic_sigmas.split(',')]

    if len(mus) == 1 and args.num_tasks > 1: mus *= args.num_tasks
    if len(sigmas) == 1 and args.num_tasks > 1: sigmas *= args.num_tasks

    assert len(mus) == args.num_tasks, "Number of --ic_mus must match --num_tasks"
    assert len(sigmas) == args.num_tasks, "Number of --ic_sigmas must match --num_tasks"

    pbm_tasks_params = []
    for i in range(args.num_tasks):
        params = {'ic_mu': mus[i], 'ic_sigma': sigmas[i]}

        if args.task_type == 'aggregation':
            params['disable_breakage'] = True
            params['disable_aggregation'] = False
        elif args.task_type == 'breakage':
            params['disable_aggregation'] = True
            params['disable_breakage'] = False
        elif args.task_type == 'combined':
            params['disable_aggregation'] = False
            params['disable_breakage'] = False
        else:
            raise ValueError(f"Unknown task_type: {args.task_type}")

        pbm_tasks_params.append(params)

        if args.ic_type == 'chen_agg':
            selected_ic_func = chen_aggregation_initial_condition
            print("--- Using Chen's Aggregation Initial Condition ---")
        elif args.ic_type == 'chen_break':
            selected_ic_func = dirac_delta_initial_condition
            print("--- Using Chen's Breakage (Dirac Approx) Initial Condition ---")
        else:
            selected_ic_func = gaussian_initial_condition
            print("--- Using Gaussian Initial Condition ---")

    print(f"Running {len(pbm_tasks_params)} tasks with type: {args.task_type}")

    set_seed(args.seed)
    VT_star, N_star, VT_ic_train, N_ic_train, VT_f_train = process_data(args, pbm_tasks_params, initial_condition_func=selected_ic_func)

    layers = [int(item) for item in args.layers.split(',')]
    layers.insert(0, 2)
    layers.append(1)

    model = iPINN(
        args=args,
        pbm_tasks_params=pbm_tasks_params,
        VT_ic_train=VT_ic_train, N_ic_train=N_ic_train, VT_f_train=VT_f_train,
        layers=layers,
        optimizer_name=args.optimizer_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        net=args.net,
        num_epochs=None,
        activation=args.activation,
        loss_style=args.loss_style,
        initial_condition_func=selected_ic_func
    )
    print("DNN architecture:")
    print(model.dnn)

    model, loss_histories = incremental_learning(
        args, model, VT_star, N_star,
        VT_ic_train, N_ic_train, VT_f_train, None, None,
        args.num_tasks, args.alpha_fc, device
    )
    validate(model, VT_star, N_star)

    if args.visualize:
        visualize_pbm_results(model, args, pbm_tasks_params, VT_star, N_star, loss_histories=loss_histories, current_task_id=args.num_tasks - 1)


if __name__ == "__main__":
    main()