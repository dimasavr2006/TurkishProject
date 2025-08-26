import torch
import numpy as np

from utils import validate
from pruning import mlp_pruning
from visualize import visualize_pbm_results
from choose_optimizer import choose_optimizer

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def incremental_learning(args, model, VT_star, N_star, VT_ic_train, N_ic_train, VT_f_train, _, __, num_tasks, alpha_fc, device):
    all_loss_histories = {}

    for task_id in range(num_tasks):
        print('=================== TASK {} ==================='.format(task_id + 1))

        model.num_learned += 1
        model.dnn.set_trainable_masks(task_id)

        print("\n--- STAGE 1: Initial Training ---")
        optimizer_adam1 = choose_optimizer("Adam", model.dnn.parameters(), args.lr, weight_decay=args.weight_decay)

        lh_adam1 = model.train(optimizer=optimizer_adam1, num_epochs=args.num_epochs_train)

        optimizer_lbfgs1 = choose_optimizer("LBFGS", model.dnn.parameters(), lr=0.1, max_iter=args.lbfgs_max_iter)
        lh_lbfgs1 = model.train(optimizer=optimizer_lbfgs1, num_epochs=args.lbfgs_max_iter)

        if args.num_tasks > 1 and task_id < args.num_tasks - 1:
            print("\n--- STAGE 2: Pruning and Retraining ---")

            print("PRUNING...")
            vt_prune = torch.FloatTensor(np.concatenate((VT_ic_train[task_id], VT_f_train[task_id]))).to(device)
            model.dnn = mlp_pruning(model.dnn, alpha_fc, vt_prune, task_id, device, start_fc_prune=0)

            model.dnn.set_trainable_masks(task_id)
            optimizer_adam2 = choose_optimizer("Adam", model.dnn.parameters(), args.lr)

            lh_adam2 = model.train(optimizer=optimizer_adam2, num_epochs=args.num_epochs_retrain)

            optimizer_lbfgs2 = choose_optimizer("LBFGS", model.dnn.parameters(), lr=0.1, max_iter=args.lbfgs_max_iter)
            lh_lbfgs2 = model.train(optimizer=optimizer_lbfgs2, num_epochs=args.lbfgs_max_iter)

            all_loss_histories[task_id] = {
                'total': lh_adam1['total'] + lh_lbfgs1['total'] + lh_adam2['total'] + lh_lbfgs2['total'],
                'ic': lh_adam1['ic'] + lh_lbfgs1['ic'] + lh_adam2['ic'] + lh_lbfgs2['ic'],
                'f': lh_adam1['f'] + lh_lbfgs1['f'] + lh_adam2['f'] + lh_lbfgs2['f'],
            }

            validate(model, VT_star, N_star, save=True)
            model.dnn.set_masks_union()
            model.dnn.save_masks(file_name=f"masks_{model.experiment_name}.pt")

            if task_id + 1 < args.num_tasks:
                model.dnn.add_mask(task_id=task_id + 1)
        else:
            all_loss_histories[task_id] = {
                'total': lh_adam1['total'] + lh_lbfgs1['total'],
                'ic': lh_adam1['ic'] + lh_lbfgs1['ic'],
                'f': lh_adam1['f'] + lh_lbfgs1['f'],
            }

        if args.visualize:
            print(f"\n--- VISUALIZING RESULTS AFTER TASK {task_id + 1} ---")
            visualize_pbm_results(
                model,
                args,
                model.pbm_params,
                VT_star,
                N_star,
                loss_histories=all_loss_histories,
                current_task_id=task_id
            )

    return model, all_loss_histories