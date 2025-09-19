# Visualization for hyperparameter tuning of GVAE, Benchmark 1

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rc('font', family = 'Times')
mpl.rcParams['mathtext.fontset'] = 'stix'

# we only vary latent dimension and Beta_KL
# other factors such as what follows are fixed
# EarlyStopping = 1000, Batch Size = 500, RNN = BIGRU, CXCYCZ = C7C8C9, Hidden: 3x80, DENSE = 80, Decoder activation = ELU

# Latent dimension, Beta KL, Acc, Reconstruction loss, Correctly reconstructed equation / 1000
hyperparams = np.array([[16, 1e-5, 0.93755, 0.00954, 199],
 [17, 1e-5, 0.93930, 0.00938, 131],
 [18, 1e-5, 0.96038, 0.00713, 322],
 [18, 1e-4, 0.94920, 0.00802, 260],
 [19, 1e-5, 0.94563, 0.00882, 183],
 [20, 1e-5, 0.94418, 0.00822, 118],
 [21, 1e-5, 0.93965, 0.01062, 202],
 [22, 1e-5, 0.94165, 0.00878, 117],
 [23, 1e-5, 0.94788, 0.00906, 196],
 [24, 1e-5, 0.95868, 0.00746, 352],
 [24, 1e-4, 0.97663, 0.00589, 691],
 [24, 1e-3, 0.95245, 0.00710, 228],
 [25, 1e-5, 0.95728, 0.00753, 262],
 [25, 1e-4, 0.95445, 0.00788, 293],
 [26, 1e-5, 0.94113, 0.00976, 208],
 [27, 1e-5, 0.93658, 0.00900, 175],
 ])

best_hyperparams = np.array([24, 1e-4, 0.97663, 0.00589, 691])

marker_types = ['o', '^', 's']
labels = [r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$']

plot_hyperparams = dict()
for i in range(hyperparams.shape[0]):
    if hyperparams[i, 1] in plot_hyperparams:
        plot_hyperparams[hyperparams[i, 1]].append([hyperparams[i, 0], hyperparams[i, 2], hyperparams[i, 3], hyperparams[i, 4]])
    else:
        plot_hyperparams[hyperparams[i, 1]] = [[hyperparams[i, 0], hyperparams[i, 2], hyperparams[i, 3], hyperparams[i, 4]]]
for key in plot_hyperparams.keys():
    plot_hyperparams[key] = np.array(plot_hyperparams[key])

fig, axs = plt.subplots(nrows=1,ncols=2, figsize=(10,4))
i = 0
for key in plot_hyperparams.keys():
    axs[0].plot(plot_hyperparams[key][:,0], plot_hyperparams[key][:,1], marker_types[i], label=r'$\beta_{KL}=$'+labels[i])
    i += 1
axs[0].plot(best_hyperparams[0], best_hyperparams[2], '1', color='black', markersize=10, label='Final model')
axs[0].legend(loc='upper left', fontsize = 10)
axs[0].set_xlabel('Latent dimension', fontsize = 10)
axs[0].set_ylabel('Accuracy', fontsize = 10)
axs[0].set_ylim([0.9,1.0])

i = 0
for key in plot_hyperparams.keys():
    axs[1].plot(plot_hyperparams[key][:,0], plot_hyperparams[key][:,3], marker_types[i],  label=labels[i])
    i += 1
axs[1].plot(best_hyperparams[0], best_hyperparams[4], '1', color='black', markersize=10, label='Final model')
axs[1].set_xlabel('Latent dimension', fontsize = 10)
axs[1].set_ylabel('Correctly reconstructed equations / 1,000', fontsize = 10)
axs[1].set_ylim([0, 1000])
#plt.savefig('Figure_8_B1_Hyperparam_Study_Appendix.png', dpi = 600, bbox_inches = 'tight')
#plt.show()