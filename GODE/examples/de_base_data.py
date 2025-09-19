# Karin Yu, 2025

# Function for ODE discovery with data with and without noise
# noise: Gaussian with std = noise_sigma
import torch.optim.lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import symengine as spe
import sys
import numpy as np
from torch.utils.data import TensorDataset
from scipy.signal import detrend

# Neural Network Class
class NeuralNet(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param,
                 regularization_exp, retrain_seed, fourier=False):
        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = torch.nn.SiLU()
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        if fourier:
            print('Using Fourier features')
            self.input_layer = torch.nn.Linear(2*self.input_dimension, self.neurons)
        else:
            self.input_layer = torch.nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = torch.nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization
        self.init_xavier()
        self.fourier_feature = fourier

    def Fourier(self, x):
        return torch.cat([x, torch.sin(x)], 1)

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        if self.fourier_feature:
            x = self.Fourier(x)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == torch.nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = torch.nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss

def CentralDiffMethod(func, h):
    return (func[2:] - func[:-2]) / (2 * h)

#if __name__ == "__main__":
# obtain data points
if not NN_LOSS_DEFINED:
    # t_val from main file
    true_sol = spe.sympify(true_solution[0])
    u_t = spe.lambdify((t), [true_sol])
    du_dt_t = spe.lambdify((t), [spe.diff(true_sol, t)])
    d2u_dt2_t = spe.lambdify((t), [spe.diff(true_sol, t, t)])
    du_dt_val = du_dt_t(t_val)
    d2u_dt2_val = d2u_dt2_t(t_val)
    u_val_noiseless = u_t(t_val)#[0] # no noise

    u_mean = np.mean(u_val_noiseless, axis = 0)
    u_std = np.sqrt(np.sum((u_val_noiseless - u_mean)**2/len(t_val), axis = 0))

    # measured data points
    u_val = torch.tensor(u_val_noiseless + noise_sigma*np.random.randn(*u_val_noiseless.shape)*u_std).float().view(-1, 1)
t_tensor = torch.tensor(t_val).float().view(-1, 1)
t_tensor.requires_grad = True

# general definition of LOSS for ENGINEERING CASES - turn on for B3
"""def LOSS(u_pred, t_tensor):
    du_dt_pred = torch.autograd.grad(u_pred, t_tensor, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    d2u_dt2_pred = torch.autograd.grad(du_dt_pred, t_tensor, grad_outputs=torch.ones_like(du_dt_pred), create_graph=True)[0]
    if DATA_TYPE == 'data':
        return torch.mean((d2u_dt2_val - d2u_dt2_pred) ** 2) + (u_pred[0]-y0[0])**2 + (du_dt_pred[0]-y0[1])**2 + 1e-3*(torch.mean(u_pred[T_ind:]**2)+torch.mean(du_dt_pred[T_ind:]**2)) #1e-3
    elif DATA_TYPE == 'data_noisy':
        return torch.mean((d2u_dt2_val_noisy - d2u_dt2_pred) ** 2) + (u_pred[0]-y0[0])**2 + (du_dt_pred[0]-y0[1])**2 + 1e-3*(torch.mean((u_pred[T_ind:]-torch.from_numpy(detrend(u_pred[T_ind:].detach().numpy(), type = 'linear')))**2) + torch.mean((du_dt_pred[T_ind:]-torch.from_numpy(detrend(du_dt_pred[T_ind:].detach().numpy(), type = 'constant')))**2))"""

# fit MLP to data
# 3x32
u_NN = NeuralNet(input_dimension=1, output_dimension=1, n_hidden_layers=3, neurons=32, regularization_param=0.0005, regularization_exp=2., retrain_seed= random_seed, fourier = FOURIER)
optimizer = torch.optim.Adam(u_NN.parameters(), lr=0.001)
if case == 'specific':
    if DATA_TYPE == 'data_noisy':
        EPOCHS = 100000 # for noisy, 100000, 120000 for SI
    else:
        EPOCHS = 50000 # for noise_less, 50000
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)
else:
    EPOCHS = 10000 # for noise_less
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.95)
train_dataset = TensorDataset(t_tensor, u_val)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
history = []
for e in tqdm(range(EPOCHS)):
    optimizer.zero_grad()
    u_pred = u_NN(t_tensor)
    if not NN_LOSS_DEFINED:
        loss = torch.mean((u_val - u_pred) ** 2)
    else:
        loss = LOSS(u_pred, t_tensor)
    loss.backward()
    optimizer.step()
    scheduler.step()
    history.append(loss.item())
u_pred = u_NN(t_tensor)

print('# Loss after {:.0f} epochs: {:.7f}'.format(EPOCHS, loss.item()))

# Obtain data evaluation points:
u_t_NN = u_NN(t_tensor)
du_dt_NN = torch.autograd.grad(u_t_NN, t_tensor,grad_outputs=torch.ones_like(u_t_NN), create_graph=True)[0]
d2u_dt2_NN = torch.autograd.grad(du_dt_NN, t_tensor,  grad_outputs=torch.ones_like(du_dt_NN), create_graph=True)[0]
d3u_dt3_NN = torch.autograd.grad(d2u_dt2_NN, t_tensor, grad_outputs=torch.ones_like(d2u_dt2_NN), create_graph=True)[0]
u_t = spe.symbols('u_t')
du_dt = spe.symbols('du_dt')
d2u_dt2 = spe.symbols('d2u_dt2')
d3u_dt3 = spe.symbols('d3u_dt3')

if NN_LOSS_DEFINED:
    plt.figure(figsize=(8, 4))
    plt.plot(t_val, u_val, label='True')
    plt.plot(t_val, u_t_NN.detach().numpy(), label='NN')
    plt.title(r'$u_t$')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(t_val, du_dt_val, label='True')
    plt.plot(t_val, du_dt_NN.detach().numpy(), label='NN')
    plt.title(r'$\dot{u}_t$')
    plt.legend()
    plt.show()

    if ORDER == 2:
        plt.figure(figsize=(8, 4))
        if 'd2u_dt2_val' in globals():
            plt.plot(t_val, d2u_dt2_val, label='True')
        if case == 'specific' and DATA_TYPE == 'data_noisy':
            plt.plot(t_val, d2u_dt2_val_noisy.detach().numpy(), '.', label='Measured', alpha = 0.3)
        elif case == 'specific' and DATA_TYPE == 'data':
            plt.plot(t_val, d2u_dt2_val.detach().numpy(), '.', label='Measured', alpha = 0.3)
        plt.plot(t_val, d2u_dt2_NN.detach().numpy(), label='NN')
        plt.legend()
        plt.show()
    print('MSE u(t):', torch.mean((u_val - u_t_NN) ** 2).item())
    print('MSE du/dt(t):', torch.mean((du_dt_val - du_dt_NN) ** 2).item())
    if 'd2u_dt2_val' in globals():
        print('MSE d2u/dt2(t):', torch.mean((d2u_dt2_val - d2u_dt2_NN) ** 2).item())