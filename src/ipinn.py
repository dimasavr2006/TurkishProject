import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
import copy

from torch.autograd import Variable

from choose_optimizer import *
from pbm_solver import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def init_weights(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class DNN(torch.nn.Module):
    def __init__(self, layers, activation, num_inputs=2, num_outputs=1, use_batch_norm=False, use_instance_norm=False):
        super(DNN, self).__init__()

        self.depth = len(layers) - 1

        if activation == 'identity':
            self.activation = torch.nn.Identity()
        elif activation == 'tanh':
            self.activation = torch.nn.functional.tanh
        elif activation == 'relu':
            self.activation = torch.nn.functional.relu
        elif activation == 'leaky_relu':
            self.activation = torch.nn.functional.leaky_relu
        elif activation == 'gelu':
            self.activation = torch.nn.functional.gelu
        elif activation == 'sin':
            self.activation = torch.sin
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )

            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i + 1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i + 1])))


        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_tasks = 0
        self.task_id = 0
        self.base_masks = self._create_masks(layers, num_inputs)

        self.tasks_masks = []
        self.add_mask(task_id=0, num_inputs=num_inputs)

        self.trainable_mask = copy.deepcopy(self.tasks_masks[0])
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])

        self.apply(init_weights)

    def _create_masks(self, layers, num_inputs=2):
        print(layers)
        masks = [torch.ones(layers[1], layers[0]), torch.ones(layers[1])]

        for l in range(1, len(layers) - 2):
            masks.append(torch.ones(layers[l + 1], layers[l]))
            masks.append(torch.ones(layers[l + 1]))

        masks.append(torch.ones(layers[-1], layers[-2]))
        masks.append(torch.ones(layers[-1]))

        return masks

    def add_mask(self, task_id, num_inputs=2, num_outputs=1):
        self.num_tasks += 1
        self.tasks_masks.append(copy.deepcopy(self.base_masks))

    def total_params(self):
        total_number = 0
        for param_name in list(self.state_dict()):
            param = self.state_dict()[param_name]
            total_number += torch.numel(param[param != 0])

        return total_number

    def total_params_mask(self, task_id):
        total_number_fc = torch.tensor(0, dtype=torch.int32)
        for mask in self.tasks_masks[task_id]:
            total_number_fc += mask.sum().int()

        return total_number_fc.item()

    def set_masks_union(self):
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        for id in range(1, self.num_tasks):
            for l in range(0, len(self.base_masks)):
                self.masks_union[l] = copy.deepcopy(1 * torch.logical_or(self.masks_union[l], self.tasks_masks[id][l]))

    def set_trainable_masks(self, task_id):
        if task_id > 0:
            for l in range(len(self.trainable_mask)):
                self.trainable_mask[l] = copy.deepcopy(1 * ((self.tasks_masks[task_id][l] + self.masks_union[l]) > 0))
        else:
            self.trainable_mask = copy.deepcopy(self.tasks_masks[task_id])

    def forward(self, x):
        u = x

        for l, layer in enumerate(list(self.layers.children())[0:-1]):
            active_weights = layer.weight * self.tasks_masks[self.task_id][2 * l].to(device)
            active_bias = layer.bias * self.tasks_masks[self.task_id][2 * l + 1].to(device)
            u = F.linear(u, weight=active_weights, bias=active_bias)
            u = self.activation(u)

        layer = list(self.layers.children())[-1]
        active_weights = layer.weight * self.tasks_masks[self.task_id][-2].to(device)
        active_bias = layer.bias * self.tasks_masks[self.task_id][-1].to(device)

        out = F.linear(u, weight=active_weights, bias=active_bias)

        return out

    def save_masks(self, file_name='net_masks.pt'):
        masks_database = {}

        for task_id in range(self.num_tasks):
            masks_database[task_id] = []
            for l in range(len(self.tasks_masks[0])):
                masks_database[task_id].append(self.tasks_masks[task_id][l])

        torch.save(masks_database, file_name)

    def load_masks(self, file_name='net_masks.pt', num_tasks=1):
        masks_database = torch.load(file_name)

        for task_id in range(num_tasks):
            for l in range(len(self.tasks_masks[task_id])):
                self.tasks_masks[task_id][l] = masks_database[task_id][l]

            if task_id + 1 < num_tasks:
                self._add_mask(task_id + 1)

        self.set_masks_union()


class iPINN():
    def __init__(self, args, pbm_tasks_params, VT_ic_train, N_ic_train, VT_f_train, layers,
                 optimizer_name, lr, weight_decay, net, num_epochs=1000,
                 activation='tanh', loss_style='mean', initial_condition_func=None):

        self.args = args
        self.pbm_params = pbm_tasks_params
        self.current_pbm_params = None

        self.set_data(VT_ic_train, N_ic_train, VT_f_train)

        self.net = net
        if self.net == 'DNN':
            self.dnn = DNN(layers, activation, num_inputs=2, num_outputs=1).to(device)
        else:
            self.dnn = torch.load(net).dnn

        num_v_quad = self.args.num_v_quad

        self.num_v_quad = num_v_quad

        self.v_quad = torch.linspace(args.v_min, args.v_max, num_v_quad, device=device)
        self.dv = self.v_quad[1] - self.v_quad[0]

        self.v_min = self.args.v_min
        self.v_max = self.args.v_max
        self.t_min = self.args.a
        self.t_max = self.args.b

        self.lb = torch.tensor([self.v_min, self.t_min], device=device).float().view(1, 2)
        self.ub = torch.tensor([self.v_max, self.t_max], device=device).float().view(1, 2)

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.optimizer_name = optimizer_name

        self.experiment_name = f"PBM_{args.activation}_alpha{args.alpha_fc}_wd{args.weight_decay}_{args.num_tasks}tasks_seed{args.seed}"
        self.num_learned = 0


        if optimizer_name == "Adam":
            self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr, (0.9, 0.999), 1e-08, weight_decay, False)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9999)
        else:
            self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr)

        self.loss_style = loss_style
        self.iter = 0

        self.initial_condition_func = initial_condition_func if initial_condition_func is not None else gaussian_initial_condition

    def set_data(self, VT_ic_train, N_ic_train, VT_f_train):
        self.v_ic, self.t_ic, self.N_ic = [], [], []
        self.v_f, self.t_f = [], []

        for task_id in range(len(VT_ic_train)):
            self.v_ic.append(torch.tensor(VT_ic_train[task_id][:, 0:1], requires_grad=True).float().to(device))
            self.t_ic.append(torch.tensor(VT_ic_train[task_id][:, 1:2], requires_grad=True).float().to(device))
            self.N_ic.append(torch.tensor(N_ic_train[task_id], requires_grad=True).float().to(device))

            self.v_f.append(torch.tensor(VT_f_train[task_id][:, 0:1], requires_grad=True).float().to(device))
            self.t_f.append(torch.tensor(VT_f_train[task_id][:, 1:2], requires_grad=True).float().to(device))

    def net_u(self, v, t):
        X = torch.cat([v, t], dim=1)
        X_normalized = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        nn_output = self.dnn(X_normalized)

        initial_condition = self.initial_condition_func(v, self.current_pbm_params)

        N_pred = initial_condition + t * nn_output
        return N_pred

    def _interpolate_on_grid(self, values_on_grid, query_points_v):
        v_q = self.v_quad
        dv = self.dv

        query_points_v = query_points_v.squeeze(-1)

        pos = (query_points_v - v_q[0]) / dv

        pos = torch.clamp(pos, 0, self.num_v_quad - 1)

        idx_floor = pos.floor().long()

        idx_ceil = torch.clamp(idx_floor + 1, 0, self.num_v_quad - 1)

        weight_ceil = (pos - idx_floor).clamp(0, 1)
        weight_floor = 1.0 - weight_ceil

        batch_idx = torch.arange(values_on_grid.shape[0], device=values_on_grid.device)

        val_floor = values_on_grid[batch_idx, idx_floor]
        val_ceil = values_on_grid[batch_idx, idx_ceil]

        interpolated_values = val_floor * weight_floor + val_ceil * weight_ceil

        return interpolated_values.unsqueeze(1)

    def calculate_pbm_residual(self, v_f, t_f):
        N_pred = self.net_u(v_f, t_f)
        dN_dt = torch.autograd.grad(
            N_pred, t_f,
            grad_outputs=torch.ones_like(N_pred),
            retain_graph=True, create_graph=True
        )[0]

        N_f = v_f.shape[0]

        v_grid_bc, t_grid_bc = torch.meshgrid(self.v_quad, t_f.squeeze(-1), indexing='ij')
        N_on_grid = self.net_u(v_grid_bc.reshape(-1, 1), t_grid_bc.reshape(-1, 1)).view(self.v_quad.shape[0], N_f).T

        kernel_agg_vals = aggregation_kernel(v_f, self.v_quad, self.current_pbm_params)
        integral_agg_death = torch.trapezoid(kernel_agg_vals * N_on_grid, self.v_quad, dim=1)
        term_agg_death = -N_pred * integral_agg_death.unsqueeze(1)

        batch_size = N_on_grid.shape[0]
        inp = N_on_grid.unsqueeze(0)
        kernel = (N_on_grid * self.dv).unsqueeze(1)
        conv_output = F.conv1d(inp, kernel, padding='same', groups=batch_size).squeeze(0)
        agg_birth_interpolated = self._interpolate_on_grid(conv_output, v_f)
        term_agg_birth = 0.5 * agg_birth_interpolated

        gamma_vals = breakage_rate_kernel(v_f, self.current_pbm_params)
        term_break_death = -gamma_vals * N_pred

        gamma_on_grid = breakage_rate_kernel(self.v_quad, self.current_pbm_params)
        kernel_daughter_vals = daughter_distribution_kernel(v_f, self.v_quad, self.current_pbm_params)
        integrand_break_birth = kernel_daughter_vals * gamma_on_grid.unsqueeze(0) * N_on_grid
        term_break_birth = torch.trapezoid(integrand_break_birth, self.v_quad, dim=1).unsqueeze(1)

        residual = dN_dt - (term_agg_birth + term_break_birth) - (term_agg_death + term_break_death)
        return residual

    def set_task(self, task_id):
        self.dnn.task_id = task_id
        self.current_pbm_params = self.pbm_params[task_id]
        return

    def rewrite_parameters(self, old_params):
        l = 0
        for param, old_param in zip(self.dnn.parameters(), old_params()):
            param.data = param.data * self.dnn.trainable_mask[l].to(device) + old_param.data * (
                    1 - self.dnn.trainable_mask[l].to(device))
            l += 1

        return

    def loss_pinn(self, return_components=False):

        loss_list, loss_u_t0_list, loss_b_list, loss_f_list = [], [], [], []

        old_grads = copy.deepcopy(self.dnn.parameters)
        for grad in old_grads():
            grad.data = torch.zeros_like(grad)

        for task_id in range(0, self.num_learned):
            self.set_task(task_id)
            loss, loss_u_t0, loss_b, loss_f = self.loss_pinn_one_task(task_id)
            loss_list.append(loss)
            loss_u_t0_list.append(loss_u_t0)
            loss_b_list.append(loss_b)
            loss_f_list.append(loss_f)

        for task_id in range(len(loss_list)):
            loss_list[task_id].backward(retain_graph=True)
            l = 0
            for param, old_grad in zip(self.dnn.layers.parameters(), old_grads()):
                param.grad.data = param.grad.data * (self.dnn.tasks_masks[task_id][l]).to(device) + old_grad.data * (
                        1 - self.dnn.tasks_masks[task_id][l]).to(device)
                old_grad.data = copy.deepcopy(param.grad.data)
                l += 1

        if return_components:
            loss_tot = torch.tensor([l.item() for l in loss_list])
            loss_u_t0_tot = torch.tensor([l.item() for l in loss_u_t0_list])
            loss_b_tot = torch.tensor([l.item() for l in loss_b_list])
            loss_f_tot = torch.tensor([l.item() for l in loss_f_list])
            return loss_tot.sum(), loss_u_t0_tot.sum(), loss_b_tot.sum(), loss_f_tot.sum()
        else:
            return sum(loss_list)

    def train_step(self, verbose=True):

        if torch.is_grad_enabled():
            self.optimizer.zero_grad()

        loss, loss_u_t0, loss_b, loss_f = self.loss_pinn()

        grad_norm = 0
        for p in self.dnn.parameters():
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        if verbose:
            if self.iter % 100 == 0:
                print(
                    'epoch %d, gradient: %.5e, loss: %.5e, loss_u0: %.5e, loss_b: %.5e, loss_f: %.5e' % (self.iter,
                                                                                                         grad_norm,
                                                                                                         loss.sum().item(),
                                                                                                         loss_u_t0.sum().item(),
                                                                                                         loss_b.sum().item(),
                                                                                                         loss_f.sum().item())
                )
            self.iter += 1

        return loss.sum().item()

    def train(self, optimizer, num_epochs):
        self.dnn.train()
        old_params = copy.deepcopy(self.dnn.parameters)
        min_loss = np.inf
        loss_history = {'total': [], 'ic': [], 'f': []}

        if isinstance(optimizer, torch.optim.Adam):
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(num_epochs / 4), gamma=0.5)
            print(f"Starting training with Adam for {num_epochs} epochs...")
            for epoch in range(num_epochs):
                def closure():
                    optimizer.zero_grad()
                    loss = self.loss_pinn()
                    loss.backward()
                    return loss

                optimizer.step(closure)
                scheduler.step()
                self.rewrite_parameters(old_params)

                if epoch % 100 == 0:
                    total_loss, ic_loss, _, f_loss = self.loss_pinn(return_components=True)
                    loss_history['total'].append(total_loss.item())
                    loss_history['ic'].append(ic_loss.item())
                    loss_history['f'].append(f_loss.item())

                    if epoch % 100 == 0:
                        print(
                            f"Epoch {epoch}: Loss={total_loss.item():.4e}, IC={ic_loss.item():.4e}, F={f_loss.item():.4e}")

                    if total_loss.item() < min_loss:
                        min_loss = total_loss.item()
                        torch.save(self.dnn.state_dict(), f"model_{self.experiment_name}.pth")

        elif isinstance(optimizer, torch.optim.LBFGS):
            print(f"Starting training with L-BFGS for max {num_epochs} iterations...")

            def closure():
                optimizer.zero_grad()
                loss = self.loss_pinn()
                loss.backward()
                return loss

            optimizer.step(closure)
            self.rewrite_parameters(old_params)

            total_loss, ic_loss, _, f_loss = self.loss_pinn(return_components=True)
            loss_history['total'].append(total_loss.item())
            loss_history['ic'].append(ic_loss.item())
            loss_history['f'].append(f_loss.item())
            print(f"After L-BFGS: Loss={total_loss.item():.4e}, IC={ic_loss.item():.4e}, F={f_loss.item():.4e}")
            torch.save(self.dnn.state_dict(), f"model_{self.experiment_name}.pth")

        else:
            raise TypeError("Optimizer not supported")

        self.dnn.load_state_dict(torch.load(f"model_{self.experiment_name}.pth"))
        return loss_history

    def loss_pinn_one_task(self, task_id):
        loss_ic = torch.tensor(0.0).to(device)

        v_f_full = self.v_f[task_id]
        t_f_full = self.t_f[task_id]

        num_f_points = v_f_full.shape[0]
        batch_size = 256

        shuffled_indices = torch.randperm(num_f_points)

        loss_f_total = 0.0

        for i in range(0, num_f_points, batch_size):
            indices = shuffled_indices[i:i + batch_size]
            v_f_batch = v_f_full[indices]
            t_f_batch = t_f_full[indices]

            residual_pred_batch = self.calculate_pbm_residual(v_f_batch, t_f_batch)

            squared_error = residual_pred_batch ** 2
            loss_f_total += torch.sum(squared_error)

        if self.loss_style == 'mean':
            loss_f = loss_f_total / num_f_points
        else:
            loss_f = loss_f_total

        loss = loss_f

        return loss, loss_ic, torch.tensor(0.0), loss_f

    def predict(self, VT_star):
        v_star = torch.tensor(VT_star[:, 0:1], requires_grad=True).float().to(device)
        t_star = torch.tensor(VT_star[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        n_star = self.net_u(v_star, t_star)
        n_star = n_star.detach().cpu().numpy()

        return n_star