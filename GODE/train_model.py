# Trains model
# ========================
# Karin Yu, 2025

__author__ = 'Karin Yu'

import h5py
import random
from parser.cfg_parser import *
from model.vae import *
import torch
import os
from tqdm import tqdm
import sys

# file to train model
if len(sys.argv) > 1:
    LATENT = int(str(sys.argv[1])) # defined in command line (e.g. python3 SolutionSolver.py 3) for case = '3'
    BETA_KL = int(sys.argv[2]) # Beta KL
else:
    LATENT = 18
    BETA_KL = 4

# modifiable parameters
BATCH = 500
if BETA_KL == 2: BETA = 0.001
elif BETA_KL == 3: BETA = 0.0001
elif BETA_KL == 4: BETA = 0.00001
Encoder_type = 'conv'
rnn_type = 'bigru'
num_hidden = 3
HIDDEN = 80
DENSE = 100
MODE = 'cpu'
CONV1 = 7
decoder_activation = 'ELU'
"""# Benchmark 1
num_data = '_11k'
data_file = 'data/data_nlode_b1'+num_data+'_train.h5'
testdata_file = 'data/data_nlode_b1'+num_data+'_test.txt'
grammar_file = 'grammar/nlode_b1.grammar'"""
"""# Benchmark 2
num_data = '_51k'
data_file = 'data/data_nlode_b2_datagen'+num_data+'_train.h5'
testdata_file = 'data/data_nlode_b2_datagen'+num_data+'_test.txt'
grammar_file = 'grammar/nlode_b2_gvae.grammar'"""
# Benchmark 3
num_data = '_41k'
data_file = 'data/data_nlode_b3_datagen'+num_data+'_train.h5'
testdata_file = 'data/data_nlode_b3_datagen'+num_data+'_test.txt'
grammar_file = 'grammar/nlode_b3_gvae.grammar'
GradientClip = 0
EARLYSTOPPING = 1000
###-----------------###

# load common arguments - change in file if needed
exec(open('grammar/common_args.py').read())

# random seed
seed = 19260817

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

### 1. load dataset
print('data_file:', data_file)
if os.path.isfile(data_file):
    print(f"File {data_file} exists.")
h5f = h5py.File(data_file, 'r')
data = h5f['data'][:]
data = torch.from_numpy(data).float()
print('One hot encoded data:', data.shape)
h5f.close()

class Model_Dataset(torch.utils.data.Dataset):
    def __init__(self, one_hot_encoding):
        self.one_hot_encoding_outputs = torch.tensor(np.array(one_hot_encoding))

    def __len__(self):
        return self.one_hot_encoding_outputs.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            idx (int): The index of the sample to fetch.
        Returns:
            tuple: (data, target) pair for the given index.
        """
        sample_one_hot_encoding_output = self.one_hot_encoding_outputs[idx]
        return sample_one_hot_encoding_output

# 2. load grammar file
grammar = Grammar(grammar_file)
dimensions = len(grammar.productions)
# attributes come into effect after decoding during sampling - here we don't sample yet

# 3. define model architecture and model_save file
params = {'hidden': HIDDEN, 'dense': DENSE, 'conv1': CONV1, 'conv2': CONV1+1, 'conv3': CONV1+2, 'rnn': rnn_type, 'num_hidden': num_hidden, 'encoder_type': Encoder_type, 'decoder_act': decoder_activation, 'mode': MODE}
if Encoder_type == 'conv':
    model_save = 'savedmodels/nlode_TEST_b3_'+params['rnn']+'_d'+str(params['dense'])+'_'+str(params['num_hidden'])+'h' + str(params['hidden']) + '_c'+str(params['conv1'])+str(params['conv2'])+str(params['conv3'])+'_L' + str(LATENT) + '_b'+str(BETA_KL)+'_'+decoder_activation+'_GC'+str(GradientClip)+'_bs' + str(BATCH) + '_ES' + str(EARLYSTOPPING)+num_data

model = AEModel(BETA, MODE, grammar)
print('Model name:', model_save)

# 4. if this results file exists already load it
if os.path.isfile(model_save+'.model'):
    model.load(grammar.productions, model_save+'.model', latent_dim = LATENT, hypers = params)
else:
    model.create(grammar.productions, max_length=MAX_LEN, latent_dim = LATENT, hypers = params)

# set to training
model.encoder.train()
model.decoder.train()

num_trainable_params_enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
num_trainable_params_dec = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_trainable_params_enc}, {num_trainable_params_dec}')

def loop_dataset(phase, ae, data_size, data_loader, optimizer=None):
    total_loss = []
    pbar = tqdm(range(0, (data_size+ (BATCH - 1) * (optimizer is None)) //BATCH), unit='batch', miniters=5000, mininterval=5)

    if phase == 'train' and optimizer is not None:
        ae.train()
    else:
        ae.eval()

    n_samples = 0
    for batch_idx, batch in enumerate(data_loader):
        # input shape: BATCH x INPUT_DIM
        # One-hot encoding shape: BATCH x MAX_LEN x DIMENSIONS
        batch_size = batch.size(0)
        true_data = np.transpose(batch.numpy(), [1, 0, 2]).astype(np.float32)
        x_inputs = np.transpose(true_data, [1, 2, 0])
        if phase == 'train':
            loss_list = ae.forward(x_inputs)
        else:
            loss_list = ae.forward(x_inputs, train=False)

        loss = loss_list[0]+loss_list[1] # total loss
        recon_loss = loss_list[0].data.cpu().numpy()
        kl = loss_list[1].data.cpu().numpy()[0]
        minibatch_loss = loss.data.cpu().numpy()[0]
        pbar.set_description(' %s loss: %0.5f recon: %0.5f kl: %0.5f' % (phase, minibatch_loss, recon_loss, kl))

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            if GradientClip > 0:
                torch.nn.utils.clip_grad_norm_(ae.parameters(), GradientClip)
            optimizer.step()
        total_loss.append(np.array([minibatch_loss, recon_loss, kl]) * batch_size)

        n_samples += batch_size
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples

    return avg_loss

if Encoder_type == 'gru' or Encoder_type == 'bigru' or Encoder_type == 'lstm':
    Dataset = Model_Dataset(data, seq_data)
else:
    Dataset = Model_Dataset(data)

# 5. train model
optimizer = torch.optim.Adam(model.autoencoder.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=500, min_lr=0.0001)
validation_split = 0.1

dataset_size = len(Dataset)
train_size = int((1-validation_split)*dataset_size)
valid_size = int(dataset_size-train_size)
train_dataset, val_dataset = torch.utils.data.random_split(Dataset, [train_size, valid_size])
print('train', train_size, 'valid', valid_size)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH, shuffle=True)

best_valid_loss = None
best_valid_loss_idx = 0
history = []
epoch = 0
while epoch < EARLYSTOPPING or (epoch - best_valid_loss_idx) < EARLYSTOPPING:

    avg_loss = loop_dataset('train', model.autoencoder, train_size, train_loader, optimizer)
    print('>>>>average \033[92mtraining\033[0m of epoch %d: loss %.5f perp %.5f kl %.5f' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

    if epoch % 1 == 0:
        valid_loss = loop_dataset('valid', model.autoencoder, valid_size, valid_loader)
        print('\033[>>>>average valid of epoch %d: loss %.5f perp %.5f kl %.5f\033[0m' % (
        epoch, valid_loss[0], valid_loss[1], valid_loss[2]))
        valid_loss = valid_loss[0]
        lr_scheduler.step(valid_loss)
        history.append([epoch, avg_loss[0], valid_loss])
        if best_valid_loss is None or valid_loss < best_valid_loss:
            best_valid_loss_idx = epoch
            best_valid_loss = valid_loss
            print('----saving to best model since this is the best valid loss so far.----')
            model.save(model_save + '.model')
    epoch += 1

grammar.add_disallowed({})

grammar_model = GrammarModel(grammar, model_save+'.model', latent_dim=LATENT, beta = 0.001, hypers = params, mode = MODE)
grammar_model.vae.encoder.eval()
grammar_model.vae.decoder.eval()

# check and encode
with open(testdata_file, 'r') as f:
    equations = f.readlines()
print('### Reconstruction of test equations:')
recon_loss_avg, acc_avg = 0, 0
cnt = 0
corr = 0
for i in range(len(equations)):
    z = grammar_model.encode([equations[i][:-1]])
    decoded, recon_loss, acc = grammar_model.decode(z, with_attr=True)
    recon_loss_avg = (cnt * recon_loss_avg + recon_loss) / (cnt + 1)
    acc_avg = (cnt * acc_avg + acc) / (cnt + 1)
    cnt += 1
    if decoded[0] == equations[i][:-1]:
        corr += 1
print('Number of correct equations: ', corr)