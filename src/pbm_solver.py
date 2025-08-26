import torch
import numpy as np

def aggregation_kernel(v: torch.Tensor, v_prime: torch.Tensor, params: dict) -> torch.Tensor:
    beta_0 = 1.0
    v_b = v.view(-1, 1)
    v_prime_b = v_prime.view(1, -1)
    return beta_0 * torch.ones_like(v_b + v_prime_b)


def breakage_rate_kernel(v: torch.Tensor, params: dict) -> torch.Tensor:
    return v ** 2


def daughter_distribution_kernel(v: torch.Tensor, v_prime: torch.Tensor, params: dict) -> torch.Tensor:
    v_b = v.view(-1, 1)
    v_prime_b = v_prime.view(1, -1)

    result = 2.0 / (v_prime_b + 1e-9)
    result = result.expand(v_b.shape[0], -1)

    mask = v_b > v_prime_b

    result = result.clone()
    result[mask] = 0.0

    return result


def generate_pbm_solution(pbm_params: dict,
                          v_domain=(0.0, 10.0), num_v=256,
                          t_domain=(0.0, 1.0), num_t=101,
                          initial_condition_func=None,
                          device='cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v_min, v_max = v_domain
    t_min, t_max = t_domain

    v_grid = torch.linspace(v_min, v_max, num_v, device=device)
    t_grid = torch.linspace(t_min, t_max, num_t, device=device)
    dv = v_grid[1] - v_grid[0]
    dt = t_grid[1] - t_grid[0]

    N_solution = torch.zeros(num_t, num_v, device=device)
    if initial_condition_func:
        N_solution[0, :] = initial_condition_func(v_grid, pbm_params)

    B_matrix = aggregation_kernel(v_grid, v_grid, pbm_params) * dv
    gamma_on_grid = breakage_rate_kernel(v_grid, pbm_params)
    alpha_matrix = daughter_distribution_kernel(v_grid, v_grid, pbm_params)
    A_matrix = alpha_matrix * gamma_on_grid.view(1, -1) * dv
    upper_tri_mask = torch.triu(torch.ones(num_v, num_v, device=device), diagonal=1)
    A_matrix *= upper_tri_mask

    for i in range(num_t - 1):
        N_current = N_solution[i, :]

        integrand_conv = (N_current * dv).unsqueeze(0).unsqueeze(0)
        conv_output = torch.nn.functional.conv1d(
            integrand_conv, integrand_conv, padding='same'
        ).squeeze(0).squeeze(0) * dv
        agg_birth_term = 0.5 * conv_output
        integral_agg_death = torch.matmul(B_matrix, N_current)
        agg_death_term = -N_current * integral_agg_death

        break_death_term = -gamma_on_grid * N_current
        break_birth_term = torch.matmul(A_matrix, N_current)

        dN_dt = agg_birth_term + agg_death_term + break_birth_term + break_death_term
        N_solution[i + 1, :] = N_current + dt * dN_dt

    return v_grid, t_grid, N_solution


def gaussian_initial_condition(v: torch.Tensor, params: dict) -> torch.Tensor:
    mu = params.get('ic_mu', 2.0)
    sigma = params.get('ic_sigma', 0.5)
    return torch.exp(-((v - mu) ** 2) / (2 * sigma ** 2))