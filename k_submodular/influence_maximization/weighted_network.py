import numpy as np
from numpy.random import RandomState
import pandas as pd
import os as os

os.getcwd()


def weighted_network(network, method, weights=None, seed=1888):
    """

    Parameters
    ----------
    network - network to add weights to
    method - weight method
    weights - dictionary of weights (from, to) -> prob
    seed - seed value

    Returns
    -------

    """
    rng = RandomState(seed)

    if weights:
        for edge in network.edges():
            network[edge[0]][edge[1]]['act_prob'] = weights[(edge[0], edge[1])]

    elif (method == "rn"):
        for edge in network.edges():
            network[edge[0]][edge[1]]['act_prob'] = rng.rand(1)[0]

    elif (method == "un"):
        for edge in network.edges():
            network[edge[0]][edge[1]]['act_prob'] = 0.1

    elif (method == "tv"):
        TV = [.1, .01, .001]
        for edge in network.edges():
            network[edge[0]][edge[1]]['act_prob'] = rng.choice(TV)

    elif (method == "wc"):

        edge_list = pd.DataFrame(list(network.edges))
        edge_list.columns = ['from', 'to']
        in_degree = pd.DataFrame(list(network.in_degree))
        in_degree.columns = ['to', 'in_degree']
        edge_list = edge_list.merge(in_degree)
        edge_list['act_prob'] = 1. / edge_list['in_degree']

        for i in range(len(edge_list)):
            network[edge_list['from'][i]][edge_list['to'][i]]['act_prob'] = edge_list['act_prob'][i]

    return network
