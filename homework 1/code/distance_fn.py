"""Homework 1: Distance functions on vectors.

Homework 1: Distance functions on vectors
Course    : Data Mining (636-0018-00L)

Auxiliary functions.

This file implements the distance functions that are invoked from the main
program.
"""
# Author: Damian Roqueiro <damian.roqueiro@bsse.ethz.ch>
# Author: Bastian Rieck <bastian.rieck@bsse.ethz.ch>

import numpy as np
import math


def manhattan_dist(v1, v2):
    # compute the Manhattan distance in numpy

    return np.sum(np.abs(v1 - v2))

def hamming_dist(v1, v2):
    v1 = np.where(v1 > 0, 1, 0)
    v2 = np.where(v2 > 0, 1, 0)
    dist = v1 != v2
    return np.sum(dist)

def euclidean_dist(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))

def chebyshev_dist(v1, v2):
    return np.max(np.abs(v1 - v2))

def minkowski_dist(v1, v2, d):
    return np.sum(np.power(np.abs(v1 - v2), d)) ** (1 / d)
