# Model for the variational autoencoder
# ================================
# Karin Yu, 2025

__author__ = 'Karin Yu'

import torch
from torch.autograd import Variable
import numpy as np
import sys
import os
from pytorch_initializer import weights_init

# file contains autoencoder models and encoder and decoder classes

# load common arguments
exec(open('grammar/common_args.py').read())

# This code has snippets from the SD-VAE implementation, mainly for Encoder, Decoder and AutoEncoder class
# Class taken from https://github.com/Hanjun-Dai/sdvae
# Author: Hanjun Dai
# License: MIT
# Multiple modifications were made and the architecture was adapted to the GVAE

class AutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder, grammar = None, beta = 0.1, mode = 'cpu',**kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.eps_std = 0.0001
        self.mode = mode
        if mode == 'gpu':
            self = self.cuda()
        if grammar:
            self._grammar = grammar

    def forward(self, inputs, x_logits = None, train = True):
        # encode
        z_mean, z_log_var = self.encoder(inputs)
        # reparametrize
        z = self.reparametrize(z_mean, z_log_var)
        # decode
        logits = self.decoder(z)
        new_logits = logits.permute(1,2,0).contiguous()

        # calculate loss
        if self.mode == 'gpu':
            recon_loss = self.bce_loss(new_logits, torch.from_numpy(inputs).cuda())
        else:
            recon_loss = self.bce_loss(new_logits, torch.from_numpy(inputs))
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)

        return (recon_loss, self.beta * torch.mean(kl_loss).unsqueeze(0))

    def reparametrize(self, mu, logvar):
        if self.training:
            eps = mu.data.new(mu.size()).normal_(0, self.eps_std)
            if self.mode == 'gpu':
                eps = eps.cuda()
            eps = Variable(eps)

            return mu + eps * torch.exp(logvar * 0.5)
        else:
            return mu

class Encoder(torch.nn.Module):
    def __init__(self, latent_dim, max_length, dimensions, hypers: dict, mode = 'cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.dimensions = dimensions

        self.encoder_type = hypers['encoder_type']

        if hypers['encoder_type'] == 'conv':
            self.conv_1 = torch.nn.Conv1d(dimensions, hypers['conv1'], hypers['conv1'])
            self.conv_2 = torch.nn.Conv1d(hypers['conv1'], hypers['conv2'], hypers['conv2'])
            self.conv_3 = torch.nn.Conv1d(hypers['conv2'], hypers['conv3'], hypers['conv3'])
            self.last_conv_size = max_length - hypers['conv1'] + 1 - hypers['conv2'] + 1 - hypers['conv3'] + 1
            self.dense_1 = torch.nn.Linear(self.last_conv_size * hypers['conv3'], hypers['dense'])

        self.z_mean_layer = torch.nn.Linear(hypers['dense'], latent_dim)
        self.z_log_var_layer = torch.nn.Linear(hypers['dense'], latent_dim)

        self.relu = torch.nn.ReLU()
        self.softplus = torch.nn.Softplus()

        self.mode = mode

        weights_init(self)

    def forward(self, x_cpu):
        # input dimension BATCH x DIMENSIONS x MAX_LEN
        if self.mode == 'cpu':
            batch_input = Variable(torch.from_numpy(x_cpu))
        else:
            batch_input = Variable(torch.from_numpy(x_cpu).cuda())
        if self.encoder_type == 'conv':
            h1 = self.relu(self.conv_1(batch_input))
            h2 = self.relu(self.conv_2(h1))
            h3 = self.relu(self.conv_3(h2))

        flatten = h3.view(x_cpu.shape[0], -1)
        h = self.dense_1(flatten)
        # shape h = BATCH x DENSE
        h = self.relu(h)
        # shape z = BATCH x LATENT_DIM
        z_mean = self.z_mean_layer(h)
        z_log_var = self.softplus(self.z_log_var_layer(h))
        return z_mean, z_log_var

class Decoder(torch.nn.Module):
    def __init__(self,  latent_dim, max_length, dimensions, hidden, num_hidden = 3, hypers = {'mode': 'cpu', 'rnn': 'gru', 'activation': 'ReLU'}):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.dimensions = dimensions
        self.latent_input = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.hidden_size = hidden
        self.rnn_type = hypers['rnn']
        if self.rnn_type == 'gru':
            self.rnn = torch.nn.GRU(self.latent_dim, hidden, num_hidden)
            self.decoded_logits = torch.nn.Linear(hidden, dimensions)
        elif self.rnn_type == 'bigru':
            self.rnn = torch.nn.GRU(self.latent_dim, hidden, num_hidden, bidirectional=True)  #
            self.decoded_logits = torch.nn.Linear(2*hidden, dimensions)  # 2 for bidirectional
        else:
            ValueError('rnn_type must be either lstm or gru')

        if hypers['decoder_act'] == 'ReLU':
            self.activation = torch.nn.ReLU()
        elif hypers['decoder_act'] == 'ELU':
            self.activation = torch.nn.ELU()

        self.mode = hypers['mode']
        weights_init(self)

    def forward(self, z):
        assert len(z.size()) == 2
        h = self.activation(self.latent_input(z))

        # format is MAX_LEN x BATCH x DIMENSIONS --> batch_first = False!
        rep_h = h.expand(self.max_length, z.size()[0], z.size()[1])  # repeat along time steps
        out, _ = self.rnn(rep_h)  # run multi-layer gru

        logits = self.decoded_logits(out)
        return logits


class AEModel():
    def __init__(self, beta = 0.1, mode = 'cpu', grammar = None):
        self.beta = beta
        self.mode = mode
        self.grammar = grammar

    def create(self,
               productions,
               max_length=5,
               latent_dim=10,
               hypers={'num_hidden': 3, 'hidden': 100, 'dense': 100, 'conv1': 2, 'conv2': 3, 'conv3': 4, 'rnn': 'gru', 'encoder_type': 'conv', 'decoder_act': 'ReLU', 'mode': 'cpu'},
               weights_file=None):
        dimensions = len(productions)
        self.hypers = hypers

        # Encoder
        self.encoder = Encoder(latent_dim, max_length, dimensions, self.hypers, mode = self.mode)

        # Decoder
        self.decoder = Decoder(latent_dim, max_length, dimensions, self.hypers['hidden'], num_hidden = self.hypers['num_hidden'], hypers = self.hypers)

        # Autoencoder
        print('... loading autoencoder')
        self.autoencoder = AutoEncoder(self.encoder, self.decoder, grammar = self.grammar, beta = self.beta, mode = self.mode)

        if weights_file:
            if self.mode == 'cpu':
                print('Only CPU')
                self.autoencoder.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))
            else:
                self.autoencoder.load_state_dict(torch.load(weights_file))
            print('Weights loaded.')

    def save(self, filename):
        torch.save(self.autoencoder.state_dict(), filename)

    def load(self, productions, weights_file, latent_dim=10, max_length=MAX_LEN,
             hypers={'num_hidden': 3, 'hidden': 100, 'dense': 100, 'conv1': 2, 'conv2': 3, 'conv3': 4, 'rnn': 'gru', 'encoder_type':'conv'}):
        print('... loading model')
        self.create(productions, max_length=max_length, weights_file=weights_file, latent_dim=latent_dim,
                    hypers=hypers)

class GrammarModel(object):
    def __init__(self, grammar, weights_file, hypers, latent_dim=56, beta = 0.001, mode = 'cpu'):
        """ Load the (trained) zinc encoder/decoder, grammar model. """
        self._grammar = grammar
        self.MAX_LEN = MAX_LEN
        print('Max length:', self.MAX_LEN)
        self.vae = AEModel(beta, mode)
        self.vae.load(self._grammar.productions, weights_file, max_length=self.MAX_LEN, latent_dim=latent_dim, hypers = hypers)
        self.mode = mode
        print('weights file:', weights_file)

    #TODO: Find out why encode and decode needs to be done one by one!!!
    def encode(self, input):
        """ Encode an equation strings into the latent space - 1 at a time!!! """
        assert type(input) == list
        # input shape to encoder is Batch size x Dimensions x Max Length
        onehot_list = []
        for s in input:
            onehot_list.append(self._grammar.create_one_hot(s, self.MAX_LEN))
        self.one_hot = np.vstack(onehot_list)
        x_inputs = np.transpose(self.one_hot, [0, 2, 1]) # change shape to Batch size x Dimensions x Max Length
        return self.vae.encoder(x_inputs)[0]

    def decode(self, z, with_attr = False, loss = True):
        """ Sample from the grammar decoder """
        assert z.ndim == 2
        unmasked = self.vae.decoder(z).reshape(-1, self.MAX_LEN, len(self._grammar.productions)).detach() # stays constant
        if self.mode == 'gpu':
            unmasked = unmasked.cpu()
        X_hat = self._grammar.sample_using_masks(unmasked.numpy(), with_attr) # check here there are many random things
        if loss:
            loss = self.vae.autoencoder.bce_loss(unmasked, torch.from_numpy(self.one_hot))
            y_true_class = torch.argmax(torch.from_numpy(self.one_hot), axis=-1)
            y_pred_class = torch.argmax(torch.from_numpy(X_hat), axis=-1)
            acc = (y_true_class == y_pred_class).sum().item()/y_true_class.shape[0]/y_true_class.shape[1]
            # Convert from one-hot to sequence of production rules
            return self._grammar.generate_sequence(X_hat), loss.item(), acc
        else:
            return self._grammar.generate_sequence(X_hat)



