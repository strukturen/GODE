# Checks VAE model by encoding and decoding test equations
# ================================
# Karin Yu, 2025

__author__ = 'Karin Yu'

from parser.cfg_parser import *
from model.vae import *
import numpy as np
import sys

# checks encoder and decoder
random_seed = 1238
np.random.seed(random_seed)

# choose model
if len(sys.argv) > 1:
    LATENT = int(str(sys.argv[1])) # defined in command line (e.g. python3 SolutionSolver.py 3) for case = '3'
    BETA_KL = int(sys.argv[2]) # number of runs
else:
    LATENT = 24
    BETA_KL = 3

EARLYSTOPPING = 1000
BATCH = 500
GradientClip = 0
num_hidden = 3
rnn = 'bigru'
HIDDEN = 80
DENSE = 80
CONV1 = 7
decoder_activation = 'ELU'
Encoder_type = 'conv'
# Benchmark 1
num_data = '_11k'
data_file = 'data/data_nlode_b1'+num_data+'_test.txt'
grammar_file = 'grammar/nlode_b1.grammar'
# Benchmark 2
"""num_data = '_51k'
data_file = 'data/data_nlode_b2_datagen'+num_data+'_test.txt'
grammar_file = 'grammar/nlode_b2_gvae.grammar'"""
# Benchmark 3
"""num_data = '_41k'
data_file = 'data/data_nlode_b3_datagen'+num_data+'_test.txt'
grammar_file = 'grammar/nlode_b3_gvae.grammar'"""
# Adjust MAX_LEN in grammar/common_args.py

# Parameters
params = hypers={'num_hidden': num_hidden, 'hidden': HIDDEN, 'dense': DENSE, 'conv1': CONV1, 'conv2': CONV1+1, 'conv3': CONV1+2, 'rnn': rnn, 'encoder_type': Encoder_type, 'decoder_act': decoder_activation, 'mode': 'cpu'}
if Encoder_type == 'conv':
    # Benchmark 1 = explicit
    grammar_weights = 'savedmodels/nlode_b1_explicit_' + params['rnn'] + '_d' + str(params['dense']) + '_' + str(params['num_hidden']) + 'h' + str(params['hidden']) + '_c' + str(params['conv1']) + str(params['conv2']) + str(params['conv3']) + '_L' + str(LATENT) + '_b' + str(BETA_KL) + '_' + decoder_activation + '_GC' + str(GradientClip) + '_bs' + str(BATCH) + '_ES' + str(EARLYSTOPPING) + num_data + '.model'
    # Benchmark 2
    #grammar_weights = 'savedmodels/nlode_b2_' + params['rnn'] + '_d' + str(params['dense']) + '_' + str(params['num_hidden']) + 'h' + str(params['hidden']) + '_c' + str(params['conv1']) + str(params['conv2']) + str(params['conv3']) + '_L' + str(LATENT) + '_b' + str(BETA_KL) + '_' + decoder_activation + '_GC' + str(GradientClip) + '_bs' + str(BATCH) + '_ES' + str(EARLYSTOPPING) + num_data + '.model'
    # Benchmark 3
    #grammar_weights = 'savedmodels/nlode_b3_'+params['rnn']+'_d'+str(params['dense'])+'_'+str(params['num_hidden'])+'h' + str(params['hidden']) + '_c'+str(params['conv1'])+str(params['conv2'])+str(params['conv3'])+'_L' + str(LATENT) + '_b'+str(BETA_KL)+'_'+decoder_activation+'_GC'+str(GradientClip)+'_bs' + str(BATCH) + '_ES' + str(EARLYSTOPPING)+num_data+'.model'
CHECKENDECODE = True

# 1. load grammar and VAE
grammar = Grammar(grammar_file)
disallowed_dict = {
}
grammar.add_disallowed(disallowed_dict)
grammar_model = GrammarModel(grammar, grammar_weights, latent_dim=LATENT, beta = 0.001, hypers = params, mode = 'cpu')
grammar_model.vae.encoder.eval()
grammar_model.vae.decoder.eval()

num_trainable_params_enc = sum(p.numel() for p in grammar_model.vae.encoder.parameters() if p.requires_grad)
num_trainable_params_dec = sum(p.numel() for p in grammar_model.vae.decoder.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_trainable_params_enc}, {num_trainable_params_dec}')

# 2. let's encode and decode some example equations
equations = []
# z: encoded latent points
# NOTE: this operation returns the mean of the encoding distribution
# if you would like it to sample from that distribution instead
# replace vae.py with: return self.vae.encoder.predict(one_hot)
recon_loss_avg, acc_avg = 0, 0
cnt = 0
for eq in equations[:]:
    z = grammar_model.encode([eq]) # z stays constant
    decoded, recon_loss, acc = grammar_model.decode(z)
    for i, s in enumerate(decoded):
        print('Original:', eq)
        print('Predicted:', s)
    recon_loss_avg = (cnt*recon_loss_avg + recon_loss)/(cnt+1)
    acc_avg = (cnt*acc_avg + acc)/(cnt+1)
    cnt += 1

print('Reconstruction loss: {:.5f}'.format(recon_loss_avg))
print('Accuracy: {:.5f}'.format(acc_avg))

if CHECKENDECODE:
    ### 3. load test dataset
    with open(data_file, 'r') as f:
        equations = f.readlines()
    print('### Reconstruction of test equations:')
    recon_loss_avg, acc_avg = 0, 0
    cnt = 0
    corr = 0
    for i in range(len(equations[:])):
        z = grammar_model.encode([equations[i][:-1]])
        decoded, recon_loss, acc = grammar_model.decode(z, with_attr = True)
        recon_loss_avg = (cnt * recon_loss_avg + recon_loss) / (cnt + 1)
        acc_avg = (cnt * acc_avg + acc) / (cnt + 1)
        cnt += 1
        if decoded[0] == equations[i][:-1]:
            corr += 1
        if i < 5:
            print('Original:', equations[i][:-1])
            print('Predicted:', decoded[0])
        if cnt % 250 == 0:
            print('Number of tested samples: {:.0f}'.format(i+1))
            print('Reconstruction loss: {:.5f}'.format(recon_loss_avg))
            print('Accuracy: {:.5f}'.format(acc_avg))
            print('Correct:', corr)
print('Total number of equations:', len(equations))
print('Total correctly reconstructed equations:', corr)