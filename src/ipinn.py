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


def init_weights(m): # просто инициализация весов
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class DNN(torch.nn.Module):
    # это нейронка адаптированная к инкрементальному обучению с помощью масок
    def __init__(self, layers, activation, num_inputs=2, num_outputs=1, use_batch_norm=False, use_instance_norm=False):
        super(DNN, self).__init__()

        self.depth = len(layers) - 1

        # выбор функции активации

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
            # тут со слоями работа

            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i + 1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i + 1])))


        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict) # вот мы сделали архитектуру сети

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_tasks = 0
        self.task_id = 0
        self.base_masks = self._create_masks(layers, num_inputs)

        self.tasks_masks = []
        self.add_mask(task_id=0, num_inputs=num_inputs)

        # тут маски и всё остальное

        self.trainable_mask = copy.deepcopy(self.tasks_masks[0])
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])

        self.apply(init_weights)

    # ну по названию всё понятно
    def _create_masks(self, layers, num_inputs=2):
        print(layers)
        masks = [torch.ones(layers[1], layers[0]), torch.ones(layers[1])]

        for l in range(1, len(layers) - 2):
            masks.append(torch.ones(layers[l + 1], layers[l]))
            masks.append(torch.ones(layers[l + 1]))

        masks.append(torch.ones(layers[-1], layers[-2]))
        masks.append(torch.ones(layers[-1]))

        return masks

    # сделано для того чтобы маски были независимы
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

    # объединяет все маски с помощью логического или
    def set_masks_union(self):
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        for id in range(1, self.num_tasks):
            for l in range(0, len(self.base_masks)):
                self.masks_union[l] = copy.deepcopy(1 * torch.logical_or(self.masks_union[l], self.tasks_masks[id][l]))

    def set_trainable_masks(self, task_id):

        """
        Определяет, какие веса можно обновлять.
        Для первой задачи (task_id = 0) можно обучать все веса.
        Для последующих задач, обучаемыми становятся веса, которые важны для текущей задачи (self.tasks_masks[task_id]) ИЛИ были важны для любой из предыдущих (self.masks_union).
         Это позволяет дообучать "старые" веса, если они нужны для новой задачи, и защищает веса, специфичные только для старых задач, от затирания
        """

        if task_id > 0:
            for l in range(len(self.trainable_mask)):
                self.trainable_mask[l] = copy.deepcopy(1 * ((self.tasks_masks[task_id][l] + self.masks_union[l]) > 0))
        else:
            self.trainable_mask = copy.deepcopy(self.tasks_masks[task_id])

    def forward(self, x): # прямой проход с применением масок
        u = x

        for l, layer in enumerate(list(self.layers.children())[0:-1]):
            # Выбирается маска для активной задачи
            active_weights = layer.weight * self.tasks_masks[self.task_id][2 * l].to(device) # поэлементное умножение
            active_bias = layer.bias * self.tasks_masks[self.task_id][2 * l + 1].to(device)
            u = F.linear(u, weight=active_weights, bias=active_bias) # линейное преобразование
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

        # конструктор

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
        # преобразует входные нампай массивы в тензоры пайторча и сохраняет их в списках разделяя по задачам
        self.v_ic, self.t_ic, self.N_ic = [], [], []
        self.v_f, self.t_f = [], []

        for task_id in range(len(VT_ic_train)):
            self.v_ic.append(torch.tensor(VT_ic_train[task_id][:, 0:1], requires_grad=True).float().to(device))
            self.t_ic.append(torch.tensor(VT_ic_train[task_id][:, 1:2], requires_grad=True).float().to(device))
            self.N_ic.append(torch.tensor(N_ic_train[task_id], requires_grad=True).float().to(device))

            self.v_f.append(torch.tensor(VT_f_train[task_id][:, 0:1], requires_grad=True).float().to(device))
            self.t_f.append(torch.tensor(VT_f_train[task_id][:, 1:2], requires_grad=True).float().to(device))

    def net_u(self, v, t):
        # функция которая преобразует входные предсказания

        X = torch.cat([v, t], dim=1)
        X_normalized = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0 # входные данные которые нормализуются к промежутку от -1 до 1

        C = 10.0
        nn_output = C * torch.tanh(self.dnn(X_normalized)) # трюк аналогичный статье чена который фиксирует потерю на начальных условиях

        initial_condition = self.initial_condition_func(v, self.current_pbm_params)
        initial_condition_log = torch.log(initial_condition + 1e-8)

        log_N_pred = initial_condition_log + t * nn_output
        N_pred = torch.exp(log_N_pred)

        return N_pred

    def _interpolate_on_grid(self, values_on_grid, query_points_v):

        """
        Для вычисления интегралов (например, свёртки в члене агрегации) решение N вычисляется на фиксированной сетке v_quad.
        Но невязка (residual) должна быть посчитана в произвольных коллокационных точках v_f.
        Эта функция позволяет получить значения с фиксированной сетки в этих произвольных точках
        """

        v_q = self.v_quad
        dv = self.dv

        query_points_v = query_points_v.squeeze(-1)

        pos = (query_points_v - v_q[0]) / dv

        pos = torch.clamp(pos, 0, self.num_v_quad - 1)

        idx_floor = pos.floor().long() # округляет позицию до ближайшего целого

        idx_ceil = torch.clamp(idx_floor + 1, 0, self.num_v_quad - 1)

        weight_ceil = (pos - idx_floor).clamp(0, 1) # вычисление весов для интерполяции
        weight_floor = 1.0 - weight_ceil

        batch_idx = torch.arange(values_on_grid.shape[0], device=values_on_grid.device)

        val_floor = values_on_grid[batch_idx, idx_floor]
        val_ceil = values_on_grid[batch_idx, idx_ceil]

        interpolated_values = val_floor * weight_floor + val_ceil * weight_ceil

        return interpolated_values.unsqueeze(1)

    def calculate_pbm_residual(self, v_f, t_f):

        # считает как раз таки невязку


        N_pred = self.net_u(v_f, t_f)

        dN_dt = torch.autograd.grad(
            N_pred, t_f,
            grad_outputs=torch.ones_like(N_pred),
            retain_graph=True, create_graph=True
        )[0]

        """
        Частная производная по времени ∂N/∂t вычисляется автоматически с помощью torch.autograd.grad
        Это и есть "магия" фреймворков с автоматическим дифференцированием, на которой строятся PINN
        """

        N_f = v_f.shape[0]

        v_grid_bc, t_grid_bc = torch.meshgrid(self.v_quad, t_f.squeeze(-1), indexing='ij')
        N_on_grid = self.net_u(v_grid_bc.reshape(-1, 1), t_grid_bc.reshape(-1, 1)).view(self.v_quad.shape[0], N_f).T # Здесь решение вычисляется на сетке интегрирования v_quad для каждого момента времени t из t_f


        # вычисление всех членов уравнения, мы это обсуждали ранее
        kernel_agg_vals = aggregation_kernel(v_f, self.v_quad, self.current_pbm_params)
        integral_agg_death = torch.trapezoid(kernel_agg_vals * N_on_grid, self.v_quad, dim=1)
        term_agg_death = -N_pred * integral_agg_death.unsqueeze(1)

        batch_size = N_on_grid.shape[0]
        inp = N_on_grid.unsqueeze(0)
        kernel = (N_on_grid * self.dv).unsqueeze(1)
        conv_output = F.conv1d(inp, kernel, padding='same', groups=batch_size).squeeze(0)
        agg_birth_interpolated = self._interpolate_on_grid(conv_output, v_f)

        # term_agg_birth = 0.5 * agg_birth_interpolated
        term_agg_birth = 0.5 * agg_birth_interpolated * (1.0 - float(self.current_pbm_params.get('disable_aggregation', False)))

        gamma_vals = breakage_rate_kernel(v_f, self.current_pbm_params)
        term_break_death = -gamma_vals * N_pred

        gamma_on_grid = breakage_rate_kernel(self.v_quad, self.current_pbm_params)
        kernel_daughter_vals = daughter_distribution_kernel(v_f, self.v_quad, self.current_pbm_params)
        integrand_break_birth = kernel_daughter_vals * gamma_on_grid.unsqueeze(0) * N_on_grid
        term_break_birth = torch.trapezoid(integrand_break_birth, self.v_quad, dim=1).unsqueeze(1)

        """
        Каждый физический процесс (рождение/смерть частиц из-за агрегации и дробления) представлен в виде члена уравнения.
        Интегралы вычисляются численно методом трапеций (torch.trapezoid).
        Интеграл свёртки в term_agg_birth хитро реализован через F.conv1d — это очень эффективный способ.
        residual: Это и есть невязка. Если бы N_pred было точным решением, residual был бы равен нулю. Цель обучения — минимизировать квадрат этой невязки
        """

        residual = dN_dt - (term_agg_birth + term_break_birth) - (term_agg_death + term_break_death)
        return residual

    def set_task(self, task_id):
        self.dnn.task_id = task_id
        self.current_pbm_params = self.pbm_params[task_id]
        return

    def rewrite_parameters(self, old_params):

        # эта функция вызывается после шага оптимизатора, чтобы "откатить" изменения в тех весах, которые не должны были меняться

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

        # цикл по всем выученным задачам
        for task_id in range(0, self.num_learned):
            self.set_task(task_id) # переключение маски на нужную
            loss, loss_u_t0, loss_b, loss_f = self.loss_pinn_one_task(task_id)
            loss_list.append(loss)
            loss_u_t0_list.append(loss_u_t0)
            loss_b_list.append(loss_b)
            loss_f_list.append(loss_f)
            # ну и рассчитанные потери добавляются в код

        for task_id in range(len(loss_list)):
            loss_list[task_id].backward(retain_graph=True) # вычисление градиентов только от потери выбранной задачи
            l = 0
            for param, old_grad in zip(self.dnn.layers.parameters(), old_grads()):
                param.grad.data = param.grad.data * (self.dnn.tasks_masks[task_id][l]).to(device) + old_grad.data * ( # в потери задачи task_id могут "течь" градиенты только к тем весам, которые активны для этой задачи
                        1 - self.dnn.tasks_masks[task_id][l]).to(device)  # к "отфильтрованному" градиенту прибавляются градиенты от предыдущих задач, которые хранятся в old_grad
                old_grad.data = copy.deepcopy(param.grad.data) # old_grad обновляется, чтобы на следующей итерации цикла (для task_id + 1) он содержал сумму градиентов от задач 0 до task_id
                l += 1

        if return_components:
            # блок для логирования
            loss_tot = torch.tensor([l.item() for l in loss_list])
            loss_u_t0_tot = torch.tensor([l.item() for l in loss_u_t0_list])
            loss_b_tot = torch.tensor([l.item() for l in loss_b_list])
            loss_f_tot = torch.tensor([l.item() for l in loss_f_list])
            return loss_tot.sum(), loss_u_t0_tot.sum(), loss_b_tot.sum(), loss_f_tot.sum()
        else:
            return sum(loss_list)

    def train_step(self, verbose=True):
        # не используется сейчас

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
        # просто подготовки для дальнейших отслеживаний
        old_params = copy.deepcopy(self.dnn.parameters)
        min_loss = np.inf
        loss_history = {'total': [], 'ic': [], 'f': []}

        if isinstance(optimizer, torch.optim.Adam):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1000)
            print(f"Starting training with Adam for {num_epochs} epochs...")

            for epoch in range(num_epochs):

                def closure():
                    optimizer.zero_grad() # без этого градиенты не будут накапливаться
                    loss = self.loss_pinn() # вычисляется потеря
                    loss.backward() # обратное распространенние ошибки и вычисление градиентов
                    return loss

                optimizer.step(closure)

                if epoch % 100 == 0:
                    total_loss, ic_loss, _, f_loss = self.loss_pinn(return_components=True)
                    scheduler.step(total_loss) # передаём потери чтобы оптимизатор пытался понять что происходит

                    loss_history['total'].append(total_loss.item())
                    loss_history['ic'].append(ic_loss.item())
                    loss_history['f'].append(f_loss.item())

                    print(
                        f"Epoch {epoch}: Loss={total_loss.item():.4e}, F={f_loss.item():.4e}, LR={optimizer.param_groups[0]['lr']:.2e}")

                    # сохранение лучшей модели
                    if total_loss.item() < min_loss:
                        min_loss = total_loss.item()
                        torch.save(self.dnn.state_dict(), f"model_{self.experiment_name}.pth")

                """
                Оптимизатор мог обновить все веса, но эта функция "откатывает" изменения для тех весов, которые не являются обучаемыми согласно trainable_mask.
                 Это гарантирует, что веса, важные для старых задач и не нужные для новой, не будут "затёрты"
                """

                self.rewrite_parameters(old_params)

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

        # в модел загружаются веса которые дали наименьшую потерю
        self.dnn.load_state_dict(torch.load(f"model_{self.experiment_name}.pth"))
        return loss_history

    def loss_pinn_one_task(self, task_id):

        # вычисляет лосс для одной задачи

        loss_ic = torch.tensor(0.0).to(device)

        v_f = self.v_f[task_id]
        t_f = self.t_f[task_id]

        residual_pred = self.calculate_pbm_residual(v_f, t_f)

        loss_f = torch.mean(residual_pred ** 2) # среднеквадратическая ошибка невязки

        loss = loss_f

        return loss, loss_ic, torch.tensor(0.0), loss_f # тут 0 потеря на начальных условиях так как мы её жёстко ограничили

    def _refine_collocation_points(self, task_id):
        print(f"\n--- RAR: Refining collocation points for task {task_id + 1} ---")
        self.dnn.eval()

        num_candidates = self.args.rar_candidates
        # тут логика рар, суть в том что создаются доп точки чтобы потом их проверить, суть в том чтобы найти точки с наибольшей ошибкой и их запихнуть в нашу выборку
        v_candidates = torch.tensor(
            np.random.rand(num_candidates, 1) * (self.v_max - self.v_min) + self.v_min,
            device=device, dtype=torch.float32, requires_grad=True
        )
        t_candidates = torch.tensor(
            np.random.rand(num_candidates, 1) * (self.t_max - self.t_min) + self.t_min,
            device=device, dtype=torch.float32, requires_grad=True
        )

        with torch.enable_grad():
            # тут как раз считаем невязку
            residuals = self.calculate_pbm_residual(v_candidates, t_candidates)

        with torch.no_grad():
            # тут просто квадрат невязки
            squared_residuals = residuals.pow(2)

        num_to_add = self.args.rar_points_to_add
        top_k_residuals, top_k_indices = torch.topk(squared_residuals.flatten(), num_to_add) # выбор самых плохих

        hard_v_points = v_candidates[top_k_indices]
        hard_t_points = t_candidates[top_k_indices]

        # присоединение сложных точек к датасету
        self.v_f[task_id] = torch.cat([self.v_f[task_id], hard_v_points.detach()], dim=0)
        self.t_f[task_id] = torch.cat([self.t_f[task_id], hard_t_points.detach()], dim=0)

        print(
            f"--- RAR: Added {num_to_add} new points. Total f-points for task {task_id + 1}: {self.v_f[task_id].shape[0]} ---")

        self.dnn.train()

    def predict(self, VT_star):
        v_star = torch.tensor(VT_star[:, 0:1], requires_grad=True).float().to(device) # первый столбец v и второй t
        t_star = torch.tensor(VT_star[:, 1:2], requires_grad=True).float().to(device)

        """
        Мы просто вызываем нашу функцию net_u, которая выполняет весь прямой проход через нейронную сеть, 
        включая нормализацию входа и применение формулы  
        n_star — это тензор PyTorch с предсказанными значениями
        """

        self.dnn.eval()
        n_star = self.net_u(v_star, t_star)
        n_star = n_star.detach().cpu().numpy()

        return n_star