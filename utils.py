#!/usr/bin/env python
# coding: utf-8

import os
import sys
from math import sqrt

from collections import OrderedDict

import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import precision_recall_curve
import logging


def get_logger0(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def initLabeled(y, p=0.2):
    # random selected the labeled instances' index
    n = len(y)
    labeledIndex = []
    labelDict = OrderedDict()
    for label in np.unique(y):
        labelDict[label] = []
    for i, label in enumerate(y):
        labelDict[label].append(i)
    for value in labelDict.values():
        #print(len(value))
        for idx in np.random.choice(value, size=int(p*len(value)), replace=False, p=None):
            labeledIndex.append(idx)
    return labeledIndex


def KA(feat1, feat2, remove_mean=True):
    """
        feat1, feat2: n x d
    """
    from numpy.linalg import norm
    if remove_mean:
        feat1 -= np.mean(feat1, axis=0, keepdims=1)
        feat2 -= np.mean(feat2, axis=0, keepdims=1)
    norm12 = norm(feat1.T.dot(feat2))**2
    norm11 = norm(feat1.T.dot(feat1))
    norm22 = norm(feat2.T.dot(feat2))
    return norm12 / (norm11 * norm22)


def compute_sim(feat_files):
    N = len(feat_files)
    sim = np.eye(N)
    for i in range(N):
        feat_i = np.load(feat_files[i])
        for j in range(i+1, N):
            feat_j = np.load(feat_files[j])
            sim[i, j] = KA(feat_i, feat_j, remove_mean=True)
            sim[j, i] = sim[i, j]
            print(i, j, sim[i, j], flush=True)
    return sim


def iterative_A(A, max_iterations=3):
    '''
    calculate the largest eigenvalue of A
    '''
    x = A.sum(axis=1)
    #k = 3
    for _ in range(max_iterations):
        temp = np.dot(A, x)
        y = temp / np.linalg.norm(temp, 2)
        temp = np.dot(A, y)
        x = temp / np.linalg.norm(temp, 2)
    return np.dot(np.dot(x.T, A), y)


def wpearson(vec_1, vec_2, weights=None, r=4):
    if weights is None:
        weights = [len(vec_1)-i for i in range(len(vec_1))]
    list_length = len(vec_1)
    weights = list(map(float, weights))
    vec_1 = list(map(float, vec_1))
    vec_2 = list(map(float, vec_2))
    if any(len(x) != list_length for x in [vec_2, weights]):
        print('Vector/Weight sizes not equal.')
        sys.exit(1)
    w_sum = sum(weights)

    # Calculate the weighted average relative value of vector 1 and vector 2.
    vec1_sum = 0.0
    vec2_sum = 0.0
    for x in range(len(vec_1)):
        vec1_sum += (weights[x] * vec_1[x])
        vec2_sum += (weights[x] * vec_2[x])	
    vec1_avg = (vec1_sum / w_sum)
    vec2_avg = (vec2_sum / w_sum)

    # Calculate wPCC
    sum_top = 0.0
    sum_bottom1 = 0.0
    sum_bottom2 = 0.0
    for x in range(len(vec_1)):
        dif_1 = (vec_1[x] - vec1_avg)
        dif_2 = (vec_2[x] - vec2_avg)
        sum_top += (weights[x] * dif_1 * dif_2)
        sum_bottom1 += (dif_1 ** 2 ) * (weights[x])
        sum_bottom2 += (dif_2 ** 2) * (weights[x])

    cor = sum_top / (sqrt(sum_bottom1 * sum_bottom2))
    return round(cor, r)
 
