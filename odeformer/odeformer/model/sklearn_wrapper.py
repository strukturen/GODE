# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import copy
import time as _time
import numpy as np
import torch
from collections import defaultdict
from odeformer.metrics import compute_metrics
from sklearn.base import BaseEstimator
import odeformer.model.utils_wrapper as utils_wrapper
from odeformer.model.mixins import PredictionIntegrationMixin
import traceback
from sklearn import feature_selection 
from odeformer.envs.generators import integrate_ode
from odeformer.envs.utils import *
import warnings
import scipy

#ADAPTED
#from .model import build_modules
from .embedders import LinearPointEmbedder, TwoHotEmbedder
from .transformer import TransformerModel
from odeformer.model.model_wrapper import ModelWrapper
from odeformer.envs.environment import FunctionEnvironment
from odeformer.envs import build_env

# TODO: remove and extract from odeformer.model
def build_modules(env, params):
    """
    Build modules.
    """
    modules = {}
    modules["embedder"] = LinearPointEmbedder(params, env)
    env.get_length_after_batching = modules["embedder"].get_length_after_batching

    modules["encoder"] = TransformerModel(
        params,
        env.float_id2word,
        is_encoder=True,
        with_output=False,
        use_prior_embeddings=True,
        positional_embeddings=params.enc_positional_embeddings,
    )
    if params.use_two_hot:
        dec_id2word = copy(env.equation_id2word)
        dec_id2word.update(env.constant_id2word)
    else:
        dec_id2word = env.equation_id2word
    if params.masked_input:
        modules["encoder"].proj = nn.Linear(params.enc_emb_dim,
                                            self.params.float_descriptor_length*params.max_dimension*len(env.float_id2word),
                                            bias=True)

    modules["decoder"] = TransformerModel(
        params,
        dec_id2word,
        is_encoder=False,
        with_output=True,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
    )

    # reload pretrained modules
    if params.reload_model != "":
        #logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model)
        for k, v in modules.items():
            assert k in reloaded
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            v.load_state_dict(reloaded[k])

    # log
    """for k, v in modules.items():
        logger.debug(f"{v}: {v}")
    for k, v in modules.items():
        logger.info(
            f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}"
        )"""

    # cuda
    if not params.cpu:
        for v in modules.values():
            v.cuda()

    return modules


def exchange_node_values(tree, dico):
    new_tree = copy.deepcopy(tree)
    for (old, new) in dico.items():
        new_tree.replace_node_value(old, new)
    return new_tree

class SymbolicTransformerRegressor(BaseEstimator, PredictionIntegrationMixin):

    def __init__(self,
                model=None,
                from_pretrained=False,
                max_input_points=10000,
                rescale=True,
                params=None,
                model_kwargs={},
                ):

        self.max_input_points = max_input_points
        self.model = model
        self.rescale = rescale
        self.params = params
        if from_pretrained:
            self.load_pretrained()
        for kwarg, val in model_kwargs.items():
            setattr(self.model, kwarg, val)

        if not self.params:
            feature_scale = 1
            time_range = 10
        else:
            feature_scale = self.params.init_scale
            time_range = self.params.time_range
        self.scaler = utils_wrapper.Scaler(time_range=[1, time_range], feature_scale=feature_scale) if self.rescale else None 

    def load_pretrained(self, mode = 'cpu'):
        import gdown
        #model_path = "odeformer.pt" # Original
        #model_path = "experiments/debug/12744567/checkpoint.pth" # 64 x 64 x 64, trained without rescaling
        #model_path = "experiments/debug/12744315/checkpoint.pth" # 64 x 128 x 128, trained without rescaling
        #model_path = "experiments/debug/13559580/checkpoint.pth" # 64 x 64 x 64, trained with rescaling
        model_path = "experiments/debug/13558930/checkpoint.pth" # 64 x 128 x 128, trained with rescaling
        if not os.path.exists(model_path):
            print(f"Downloading pretrained model and saving to {model_path}")
            #id = "18CwlutaFF_tAOObsIukrKVZMPmsjwNwF"
            id = "1L_UZ0qgrBVkRuhg5j3BQoGxlvMk_Pm1W"
            url = "https://drive.google.com/uc?id="+id
            gdown.download(url, model_path, quiet=False)
        else:
            print(f"Found pretrained model at {model_path}")
        # ADAPTED
        if mode == 'gpu':
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))
        if model_path[-3:] == ".pt":
            print("Loaded pretrained model")
            self.model = model
        else: #.pth from checkpoint
            # TODO: understand whether this is the newest model?
            env = build_env(self.params)
            modules = build_modules(env, self.params)
            requires_grad = False
            for k, v in modules.items():
                weights = model[k]
                v.load_state_dict(weights)
                v.requires_grad = requires_grad
            self.model = ModelWrapper(env, modules['embedder'], modules['encoder'], modules['decoder'], beam_type = 'sampling')
        if False:
            num_trainable_params_emb = sum(p.numel() for p in self.model.embedder.parameters() if p.requires_grad)
            num_trainable_params_enc = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
            num_trainable_params_dec = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)
            print('Number of parameters in embedder:', num_trainable_params_emb)
            print('Number of parameters in encoder:', num_trainable_params_enc)
            print('Number of parameters in decoder:', num_trainable_params_dec)
            print(self.model)
        # until here

    def set_args(self, args={}):
        for arg, val in args.items():
            assert hasattr(self, arg), "{} arg does not exist".format(arg)
            setattr(self, arg, val)

    def set_model_args(self, args={}):
        for arg, val in args.items():
            assert hasattr(self.model, arg), "{} arg does not exist".format(arg)
            setattr(self.model, arg, val)

    def print(self, dataset_number=None, n_predictions=1):
        n_datasets = len(self.predictions)
        if dataset_number is None:
            assert n_datasets==1, "Need to specify dataset number"
            dataset_number = 0
        else: 
            assert dataset_number<n_datasets, "Dataset {} does not exist".format(dataset_number)
        for candidate in self.predictions[dataset_number][:n_predictions]:
            equations = candidate.infix().split('|')
            for dim, equation in enumerate(equations):
                print(f"x_{dim}' = {equation.lstrip().rstrip()}")

    def fit(
        self,
        times,
        trajectories,
        sort_candidates=True,
        sort_metric="snmse",
        rescale=None,
        verbose=False,
    ):
        if not rescale: rescale = self.rescale
        self.rescale = rescale

        assert not (self.model.average_trajectories and self.rescale), "Cannot average trajectories and rescale at the same time"
        #assert not (self.params is None and self.rescale), "Need to know the time and feature range to rescale to"

        if not isinstance(times, list):
            times = [times]
            trajectories = [trajectories]
        n_datasets = len(times)
        
        # rescale time and features
        scale_params = {}
        if self.scaler is not None:
            scaled_times = []
            scaled_trajectories = []
            for i, (time, trajectory) in enumerate(zip(times, trajectories)):
                scaled_time, scaled_trajectory = self.scaler.fit_transform(time, trajectory)
                scaled_times.append(scaled_time)
                scaled_trajectories.append(scaled_trajectory)
                scale_params[i] = self.scaler.get_params()
        else:
            scaled_times = times.copy()
            scaled_trajectories = trajectories.copy()

        #print(scaled_times, scaled_trajectories)

        # permute trajectories so that when bagging the model doesn't get chunks
        for i, (scaled_time, scaled_trajectory) in enumerate(zip(scaled_times, scaled_trajectories)):
            permutation = np.random.permutation(len(scaled_time))
            scaled_times[i] = scaled_time[permutation]
            scaled_trajectories[i] = scaled_trajectory[permutation] # Original, for testing
            #scaled_trajectories[i] = scaled_trajectory[:,permutation]  #ADAPTED, for training of no rescaling

        # split into bags of size max_input_points
        inputs, inputs_ids = [], []
        for seq_id in range(len(scaled_times)):
            for seq_l in range(len(scaled_times[seq_id])):
                y_seq =  scaled_trajectories[seq_id] # Original, for testing
                #y_seq = scaled_trajectories[seq_id][0] #ADAPTED, for training of no rescaling
                if len(y_seq.shape) == 1:
                    y_seq = np.expand_dims(y_seq,-1)
                if seq_l % self.max_input_points == 0:
                    inputs.append([])
                    inputs_ids.append(seq_id)
                inputs[-1].append([scaled_times[seq_id][seq_l], y_seq[seq_l]])
            # inputs.append([])
            # inputs_ids.append(seq_id)

        # inputs more or less = t_i, x_i for each data point

        # Forward transformer
        # yuk: predicts a bunch of equations
        forward_time=_time.time()
        outputs = self.model(inputs)  ##Forward transformer: returns predicted functions

        if verbose: print("Finished forward in {} secs".format(_time.time()-forward_time))

        all_candidates = defaultdict(list)
        assert len(inputs) == len(outputs), "Problem with inputs and outputs"
        for i in range(len(inputs)):
            input_id = inputs_ids[i]
            candidates = outputs[i]
            if not candidates: all_candidates[input_id].append(None)
            for candidate in candidates:
                #ADAPTED - removed scaling for training -> bad?
                if self.scaler is not None:
                    candidate = self.scaler.rescale_function(self.model.env, candidate, *scale_params[input_id])
                    try: candidate = self.model.env.simplifier.simplify_tree(candidate)
                    except: pass
                all_candidates[input_id].append(candidate)
        #assert len(all_candidates.keys())==n_datasets
    
        if sort_candidates:
            for input_id in all_candidates.keys():
                all_candidates[input_id] = self.sort_candidates(times[input_id], trajectories[input_id], all_candidates[input_id], metric=sort_metric, verbose=verbose)

        # yuk: re-scaled equations
        self.predictions = all_candidates

        return all_candidates
    
    def predict(self, times, y0):
        if not isinstance(times, list):
            times = [times]
            y0 = [y0]
        n_datasets = len(times)
        assert len(y0)==n_datasets, "Need to provide initial conditions for each dataset"
        
        predictions = []
        for i in range(n_datasets):
            candidates = self.predictions[i]
            prediction = self.integrate_prediction(times[i], y0[i], prediction=candidates[0])
            predictions.append(prediction)

        if len(predictions)==1: predictions = predictions[0]

        return predictions

    @torch.no_grad()
    def evaluate_tree(self, tree, times, trajectory, metric):
        earliest = np.argmin(times)
        try: pred_trajectory = self.integrate_prediction(times, trajectory[earliest], prediction=tree)
        except: return np.nan
        metrics = compute_metrics(pred_trajectory, trajectory, predicted_tree=tree, metrics=metric)
        return metrics[metric][0]

    @torch.no_grad()
    def sort_candidates(self, times, trajectory, candidates, metric="snmse", verbose=False):
        if "r2" in metric: 
            descending = True
        else: 
            descending = False
        scores = []
        for candidate in candidates:
            score = self.evaluate_tree(candidate, times, trajectory, metric)
            if math.isnan(score): 
                score = -np.infty if descending else np.infty
            scores.append(score)
        sorted_idx = np.argsort(scores)  

        if descending: sorted_idx= list(reversed(sorted_idx))
        candidates = [candidates[i] for i in sorted_idx]

        scores = [scores[i] for i in sorted_idx]

        if verbose: 
            print(scores, candidates)
            for score, candidate in zip(scores, candidates):
                print(f'{score}:{candidate}')

        return candidates
