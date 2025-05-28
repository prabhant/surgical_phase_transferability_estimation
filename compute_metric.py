#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import numpy as np
import re
import json
import time

from metrics import LEEP, NLEEP, LogME_Score, SFDA_Score, PARC_Score,KFDA_Score,MyFDA_Score,Energy_Score,LDA_Score,PAC_Score,ft_int_gpu, Transrate, ft_wg_dist_gpu, NCTI_Score
# from gbc import get_gbc_score
from hscore import getHscore

def save_score(score_dict, fpath):
    with open(fpath, "w") as f:
        # write dict 
        json.dump(score_dict, f)


parser = argparse.ArgumentParser(description='Calculate transferability score.')
parser.add_argument('-d', '--dataset', type=str, default='voc2007', 
                    help='name of the dataset to evaluate on')
parser.add_argument('-me', '--metric', type=str, default='energy', 
                    help='name of the method for measuring transferability')   
parser.add_argument('--output-dir', type=str, default='./results_metrics/', 
                    help='dir of output score')

args = parser.parse_args()   

fpath = './results_metrics/{args.metric}'
if args.dataset == 'a':
    dataset = 'AutoLapro'
if args.dataset == 'r':
    dataset = 'RAMIE'


models = os.listdir(dataset)
score_dict_model = dict()
for model in models:
    print(f'computing score for model = {model}')
    folds = os.listdir(f'{dataset}/{model}')
    fold_list = [re.match(r'^[^_]+', item).group() for item in folds]
    score_dict_fold = dict()
    for fold in fold_list:
        model_npy_feature = f'{dataset}/{model}/{fold}_features.npy'
        model_npy_label = f'{dataset}/{model}/{fold}_labels.npy'
        X_features, y_labels = np.load(model_npy_feature), np.load(model_npy_label)
        if args.metric == 'int':
            if X_features.shape[0] > 6000 and model == 'endovit':
                pass
            else:
                print(f'fold name {fold} model name {model}, fold dimansions {X_features.shape}')
                score_dict_fold[fold] = ft_int_gpu(X_features, y_labels)
                print(f'fold {fold} computed {args.metric} score {score_dict_fold[fold]}')
        elif args.metric == 'logme':
            score_dict_fold[fold] = LogME_Score(X_features, y_labels)
            print(f'fold {fold} computed logme score {score_dict_fold[fold]}')
        elif args.metric == 'energy':
            score_dict_fold[fold] = Energy_Score(X_features,0.5,'bot').tolist()
            print(f'fold {fold} computed logme score {score_dict_fold[fold]}')
        elif args.metric == 'sfda': # Unstable computations for networks
            score_dict_fold[fold] = SFDA_Score(X_features, y_labels)
        elif args.metric == 'transrate': # Unstable computations for networks
            try:
                score_dict_fold[fold] = Transrate(X_features, y_labels)
                print(f'worked for {model} and fold {fold}')
            except:
                print(f'didnt work for model  {model} fold {fold}')
                pass
        elif args.metric == 'wg_dist':
            score_dict_fold[fold] = -ft_wg_dist_gpu(X_features, y_labels)
        elif args.metric == 'neg_wg_dist':
            score_dict_fold[fold] = -ft_wg_dist_gpu(X_features, y_labels)
        elif args.metric == 'nleep':           
            ratio = 5 
            score_dict_fold[fold] = NLEEP(X_features, y_labels, component_ratio=ratio)
        elif args.metric == 'hscore':
            score_dict_fold[fold] = float(getHscore(X_features, y_labels))
            # breakpoint()
        elif args.metric == 'gbc':
            score_dict_fold[fold] = get_gbc_score(X_features, y_labels)


    score_dict_model[model] = score_dict_fold

print(score_dict_model)
save_score(score_dict_model, f'{args.metric}_score_{dataset}.json')


