# CFG parser for grammar
# ================================
# Karin Yu, 2025

__author__ = 'Karin Yu'

# modify tokenize to fit selected CFG

import nltk
import numpy as np
import re
from sortedcontainers import SortedList

# This code has snippets from the SD-VAE implementation
# Class taken from https://github.com/Hanjun-Dai/sdvae
# Author: Hanjun Dai
# License: MIT

# Modifications were made:
# - accept probabilistic CFG
# - individual tokenization such as 'sin', 'cos', 'exp',...
# - adding dictionaries to disallow certain nested functions
# - multiple functions on parsing, sampling with masks etc. were added

class Grammar(object):
    def __init__(self, filepath=None, pcfg = False):
        self.pcfg = pcfg
        if filepath:
            self.load(filepath)

    def load(self, filepath):
        cfg_string = ''.join(list(open(filepath).readlines()))

        # parse from nltk
        if self.pcfg:
            cfg_grammar = nltk.PCFG.fromstring(cfg_string)
            self.productions = cfg_grammar.productions()
            self.prod_probs = np.array([prod.prob() for prod in self.productions])
        else:
            cfg_grammar = nltk.CFG.fromstring(cfg_string)
            self.productions = cfg_grammar.productions()
        self.cfg_parser = nltk.ChartParser(cfg_grammar)

        self.dim = len(self.productions)
        print('Number of production rules: ', self.dim)

        # our info for rule matching
        self.head_to_rules = head_to_rules = {}
        self.valid_tokens = valid_tokens = set()
        rule_ranges = {}
        total_num_rules = 0
        first_head = None
        for line in cfg_string.split('\n'):
            if len(line.strip()) > 0:
                head, rules = line.split('->')
                head = nltk.grammar.Nonterminal(head.strip())    # remove space
                rules = [_.strip() for _ in rules.split('|')]    # split and remove space
                rules = [
                    tuple([nltk.grammar.Nonterminal(_) if not _.startswith("'") else _[1:-1] for _ in rule.split()])
                    for rule in rules
                ]
                head_to_rules[head] = rules

                for rule in rules:
                    for t in rule:
                        if isinstance(t, str):
                            valid_tokens.add(t)

                if first_head is None:
                    first_head = head

                rule_ranges[head] = (total_num_rules, total_num_rules + len(rules))
                total_num_rules += len(rules)

        self.start_index = first_head

        self.rule_ranges = rule_ranges
        self.total_num_rules = total_num_rules

        all_lhs = [a.lhs().symbol() for a in self.productions]
        self.lhs_list = []
        for a in all_lhs:
            if a not in self.lhs_list:
                self.lhs_list.append(a)

        self._prod_map = {}
        for ix, prod in enumerate(self.productions):
            self._prod_map[prod] = ix

        self._lhs_map = {}
        for ix, lhs in enumerate(self.lhs_list):
            self._lhs_map[lhs] = ix

        self.masks = np.zeros((len(self.lhs_list), self.dim))
        count = 0
        # all_lhs.append(0)
        for sym in self.lhs_list:
            is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1, -1)
            # pdb.set_trace()
            self.masks[count] = is_in
            count = count + 1

        index_array = []
        for i in range(self.masks.shape[1]):
            index_array.append(np.where(self.masks[:, i] == 1)[0][0])
        self.ind_of_ind = np.array(index_array)

        self.disallowed_dict = dict()

    def tokenize(self, sent):
        s = sent
        funcs = ['sin', 'cos', 'exp', 'tan', 'log', 'diff']
        for fn in funcs:
            s = s.replace(fn + '(', fn + ' ')
        #s = s.replace('diff', 'Derivative')
        s = re.sub(r'([^a-z ])', r' \1 ', s)
        for fn in funcs:
            s = s.replace(fn, fn + ' (')
        return s.split()

    def add_disallowed(self, disallowed_dict):
        print('New dictionary added')
        self.disallowed_dict = disallowed_dict
        terminal_values, self.terminals_keys = get_terminals_from_dict(disallowed_dict)
        self.associated_rules_values = self.get_associated_rules(terminal_values)

    def parse(self, sentence):
        assert isinstance(sentence, str)
        tokens = self.tokenize(sentence)
        return list(self.cfg_parser.parse(tokens))

    def get_sequence(self, sentence):
        # get sequence of production rules as indices
        parse_trees = self.parse(sentence)
        productions_seq = [tree.productions() for tree in parse_trees]
        indices = [np.array([self._prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
        return indices

    def create_one_hot(self, sentence, max_length):
        indices = self.get_sequence(sentence)
        one_hot = np.zeros((len(indices), max_length, self.dim), dtype=np.float32)
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions), indices[i]] = 1.
            one_hot[i][np.arange(num_productions, max_length), -1] = 1.
        return one_hot

    def generate_sequence(self, X_hat):
        prod_seq = [[self.productions[X_hat[index, t].argmax()]
                     for t in range(X_hat.shape[1])]
                    for index in range(X_hat.shape[0])]
        return [prods_to_eq(prods) for prods in prod_seq]

    def get_associated_rules(self, terminals: list):
        associated_rules = dict()
        for terminal in terminals:
            associated_rules[terminal] = []
            for production in self.productions:
                if terminal in (production.rhs()):
                    associated_rules[terminal].append(production)
        return associated_rules

    def attribute_mask(self, mask, nest_tracker):
        # mask is a dictionary with keys being the non-terminals and values being the associated rules
        # nest_tracker is a dictionary with keys being the non-terminals and values being the number of times the non-terminal has been nested
        # the function returns a mask with the same shape as the input mask, but with the values of the disallowed non-terminals set to 0
        # the function also returns the updated nest_tracker
        for ix in range(len(mask)):
            for nt in nest_tracker[ix]:
                if nt[1] in self.disallowed_dict:
                    for lhs in self.disallowed_dict[nt[1]]:
                        for p in self.associated_rules_values[lhs]:
                            ind = self.productions.index(p)
                            mask[ix][ind] = 0

    def sample_using_masks(self, unmasked, with_attr = False):
        eps = 1e-100
        X_hat = np.zeros_like(unmasked)

        # Create a stack for each input in the batch
        S = np.empty((unmasked.shape[0],), dtype=object)
        for ix in range(S.shape[0]):
            S[ix] = [str(self.start_index)]

        if with_attr:
            nest_tracker = [SortedList() for _ in range(unmasked.shape[0])]
            nest_tracker_indices = [SortedList() for _ in range(unmasked.shape[0])]

        # Loop over time axis, sampling values and updating masks
        for t in range(unmasked.shape[1]):
            next_nonterminal = [self._lhs_map[pop_or_nothing(a)] for a in S]
            mask = self.masks[next_nonterminal]
            if with_attr:
                self.attribute_mask(mask, nest_tracker)
            if self.pcfg:
                masked_output = self.prod_probs*(np.exp(unmasked[:, t, :]) * mask + eps)
            else:
                masked_output = np.exp(unmasked[:, t, :]) * mask + eps
            masked_output = masked_output / np.sum(masked_output, axis=-1, keepdims=True)
            # Check which sampler works better
            # Gumbel
            #sampled_output = np.argmax(np.random.gumbel(size=masked_output.shape) + np.log(masked_output), axis=-1)
            # Random
            #sampled_output = np.array([np.random.choice(masked_output.shape[-1], p=masked_output[i]) for i in range(masked_output.shape[0])])
            # Deterministic
            sampled_output = np.argmax(np.log(masked_output), axis=-1)
            X_hat[np.arange(unmasked.shape[0]), t, sampled_output] = 1.0
            # Identify non-terminals in RHS of selected production, and
            # push them onto the stack in reverse order
            rhs = [filter(lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None'),
                          self.productions[i].rhs())
                   for i in sampled_output]
            for ix in range(S.shape[0]):
                S[ix].extend(list(map(str, rhs[ix]))[::-1])
                # Check nesting
                if with_attr:
                    for ix in range(len(sampled_output)):
                        if len(nest_tracker[ix]) > 0:
                            ind = nest_tracker_indices[ix].bisect_right(len(S[ix]))
                            nest_tracker[ix] = SortedList(nest_tracker[ix][:ind])
                            nest_tracker_indices[ix] = SortedList(nest_tracker_indices[ix][:ind])
                        for term_key in self.terminals_keys:
                            if term_key in self.productions[sampled_output[ix]].rhs():
                                nest_tracker[ix].add([len(S[ix]), term_key, dict()])
                                nest_tracker_indices[ix].add(len(S[ix]))
        return X_hat

def pop_or_nothing(S):
    try: return S.pop()
    except: return 'Nothing'

def reconstruct(root):
    return ''.join(root.leaves())

def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                break
    try:
        return ''.join(seq)
    except:
        return ''

def get_terminals_from_dict(dictionary: dict):
    restricted_keys = set()
    restricted_values = set()
    for key in dictionary:
        restricted_keys.add(key)
        for value in dictionary[key]:
            restricted_values.add(value)
    return list(restricted_values), list(restricted_keys)