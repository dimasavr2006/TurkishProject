import torch
import numpy as np

def aggregation_kernel(v: torch.Tensor, v_prime: torch.Tensor, params: dict) -> torch.Tensor:
    # входные тензоры объёмов
    # функция того как
    """
     описывает частоту, с которой частица объёма v слипается (агрегирует) с частицей объёма v'
    """
    if params.get('disable_aggregation', False):
        return torch.zeros_like(v.view(-1, 1) + v_prime.view(1, -1))

    beta_0 = params.get('beta_0', 1.0) # константа скорости агрегации
    v_b = v.view(-1, 1)
    v_prime_b = v_prime.view(1, -1)
    return beta_0 * torch.ones_like(v_b + v_prime_b)
    # тут и далее частично операции с матрцицами которые я не знаю


def breakage_rate_kernel(v: torch.Tensor, params: dict) -> torch.Tensor:
    # описывает частоту (скорость), с которой частица объёма v распадается на более мелкие
    if params.get('disable_breakage', False):
        return torch.zeros_like(v)

    gamma_0 = params.get('gamma_0', 1.0)
    power = params.get('breakage_power', 2.0) # степень с которой скорость зависсит от объёма
    return gamma_0 * (v ** power)


def daughter_distribution_kernel(v: torch.Tensor, v_prime: torch.Tensor, params: dict) -> torch.Tensor:
    #  описывает распределение "дочерних" частиц. Она отвечает на вопрос: "При распаде частицы объёма v', какая доля осколков будет иметь объём v
    if params.get('disable_breakage', False):
        return torch.zeros(v.view(-1, 1).shape[0], v_prime.view(1, -1).shape[1], device=v.device)

    result = 2.0 / (v_prime.view(1, -1) + 1e-9) # это как раз то что и просиходит
    result = result.expand(v.view(-1, 1).shape[0], -1)

    mask = v.view(-1, 1) > v_prime.view(1, -1)
    result = result.clone()
    result[mask] = 0.0 # большие частицы не проходят проверку

    return result

def generate_pbm_solution(pbm_params: dict,
                          v_domain=(0.0, 10.0), num_v=256,
                          t_domain=(0.0, 1.0), num_t=101,
                          initial_condition_func=None,
                          device='cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v_min, v_max = v_domain
    t_min, t_max = t_domain

    v_grid = torch.linspace(v_min, v_max, num_v, device=device) # тут создаются сетки для объёма и времени
    t_grid = torch.linspace(t_min, t_max, num_t, device=device)
    dv = v_grid[1] - v_grid[0]
    dt = t_grid[1] - t_grid[0] # это шаги по объёму и времени (объём дл численного интегрирования, а время для просто как шаг)

    N_solution = torch.zeros(num_t, num_v, device=device) # матрица для сохранения решения
    if initial_condition_func:
        N_solution[0, :] = initial_condition_func(v_grid, pbm_params) # заполняется начальным распределением

    # шаг с предвычисленим матриц, часть которую я не очень понимаю
    B_matrix = aggregation_kernel(v_grid, v_grid, pbm_params) * dv
    gamma_on_grid = breakage_rate_kernel(v_grid, pbm_params)
    alpha_matrix = daughter_distribution_kernel(v_grid, v_grid, pbm_params)
    A_matrix = alpha_matrix * gamma_on_grid.view(1, -1) * dv
    upper_tri_mask = torch.triu(torch.ones(num_v, num_v, device=device), diagonal=1)
    A_matrix *= upper_tri_mask

    for i in range(num_t - 1): # цикл по времени
        N_current = N_solution[i, :] # распределение частиц на текущем шаге

        # первое это интеграл свёртки
        agg_birth_term = 0.5 * torch.nn.functional.conv1d(N_current.unsqueeze(0).unsqueeze(0), N_current.unsqueeze(0).unsqueeze(0), padding='same').squeeze(0).squeeze(0) * dv * (1-float(pbm_params.get('disable_aggregation', False)))
        # интеграл который в дискретном виде произведение матрицыы на вектор
        agg_death_term = -N_current * torch.matmul(B_matrix, N_current)

        # смерть частиц от дробления
        break_death_term = -gamma_on_grid * N_current
        # рождение от дробления, изначально тоже интеграл
        break_birth_term = torch.matmul(A_matrix, N_current)

        dN_dt = agg_birth_term + agg_death_term + break_birth_term + break_death_term # сумма всех членов

        N_solution[i + 1, :] = N_current + dt * dN_dt # тут просто новое значение это старое + скорость изменения на шаг

    return v_grid, t_grid, N_solution

def gaussian_initial_condition(v: torch.Tensor, params: dict) -> torch.Tensor:
    mu = params.get('ic_mu', 2.0)
    sigma = params.get('ic_sigma', 0.5)
    return torch.exp(-((v - mu) ** 2) / (2 * sigma ** 2))


def chen_aggregation_initial_condition(v: torch.Tensor, params: dict) -> torch.Tensor:
    v0 = params.get('ic_mu', 2.0)
    N0 = params.get('ic_N0', 1.0)
    v0 = v0 + 1e-9

    return (N0 / v0) * (v / v0) * torch.exp(-v / v0)


def dirac_delta_initial_condition(v: torch.Tensor, params: dict) -> torch.Tensor:
    mu = params.get('ic_mu', 1.0)
    sigma = params.get('ic_sigma_approx', 0.05)

    return torch.exp(-((v - mu) ** 2) / (2 * sigma ** 2))