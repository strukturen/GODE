sol = solve_ode(true_eq, t_low, t_high, fs, y0, order = ORDER, CHECK_SUCCESS = False, method = 'RK45')

NN_LOSS_DEFINED = True

t_val = torch.tensor(sol[0]).float().view(-1, 1)
u_val_noiseless = torch.tensor(sol[1]).float().view(-1, 1)
du_dt_val = torch.tensor(sol[2]).float().view(-1, 1)

if DATA_TYPE == 'data_noisy':
    u_t_mean = torch.mean(u_val_noiseless, dim=0)
    u_t_std = torch.sqrt(torch.sum((u_val_noiseless - u_t_mean) ** 2 / len(t_val), dim=0))
    print('Noise level:', noise_sigma)
    # measured data points
    u_val = u_val_noiseless + noise_sigma * torch.randn(*u_val_noiseless.shape) * u_t_std
else:
    u_val = u_val_noiseless

# Turn off function in de_base_data.py
def LOSS(u_pred, t_tensor):
    if DATA_TYPE == 'data':
        return torch.mean((u_val_noiseless - u_pred) ** 2)
    if DATA_TYPE == 'data_noisy':
        return torch.mean((u_val - u_pred) ** 2)