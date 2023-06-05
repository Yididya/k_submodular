"""
Implement independent cascade model
"""
#!/usr/bin/env python
#    Copyright (C) 2004-2010 by
#    Hung-Hsuan Chen <hhchen@psu.edu>
#    All rights reserved.
#    BSD license.
#    NetworkX:http://networkx.lanl.gov/.
__author__ = """Hung-Hsuan Chen (hhchen@psu.edu)"""

import copy
import networkx as nx
import random
import pickle

__all__ = ['independent_cascade']

import numpy as np


def independent_cascade(G, seeds, steps=0, weighted_graph=True, copy_graph=False):
  """Return the active nodes of each diffusion step by the independent cascade
  model

  Parameters
  -----------
  G : graph
    A NetworkX graph
  seeds : list of nodes
    The seed nodes for diffusion
  steps: integer
    The number of steps to diffuse.  If steps <= 0, the diffusion runs until
    no more nodes can be activated.  If steps > 0, the diffusion runs for at
    most "steps" rounds

  Returns
  -------
  layer_i_nodes : list of list of activated nodes
    layer_i_nodes[0]: the seeds
    layer_i_nodes[k]: the nodes activated at the kth diffusion step

  Notes
  -----
  When node v in G becomes active, it has a *single* chance of activating
  each currently inactive neighbor w with probability p_{vw}

  Examples
  --------
  >>> DG = nx.DiGraph()
  >>> DG.add_edges_from([(1,2), (1,3), (1,5), (2,1), (3,2), (4,2), (4,3), \
  >>>   (4,6), (5,3), (5,4), (5,6), (6,4), (6,5)], act_prob=0.2)
  >>> layers = networkx_addon.information_propagation.independent_cascade(DG, [6])

  References
  ----------
  [1] David Kempe, Jon Kleinberg, and Eva Tardos.
      Influential nodes in a diffusion model for social networks.
      In Automata, Languages and Programming, 2005.
  """
  if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
      raise Exception( \
          "independent_cascade() is not defined for graphs with multiedges.")

  # make sure the seeds are in the graph
  for s in seeds:
    if s not in G.nodes():
      raise Exception("seed", s, "is not in graph")

  # change to directed graph
  if copy_graph:
    if not G.is_directed():
      DG = G.to_directed()
    else:
      DG = copy.deepcopy(G)
  else:
    DG = G

  # init activation probabilities
  if not weighted_graph:
    for e in DG.edges():
      if 'act_prob' not in DG[e[0]][e[1]]:
        DG[e[0]][e[1]]['act_prob'] = 0.1
      elif DG[e[0]][e[1]]['act_prob'] > 1:
        raise Exception("edge activation probability:", \
            DG[e[0]][e[1]]['act_prob'], "cannot be larger than 1")

  # perform diffusion
  A = copy.deepcopy(seeds)  # prevent side effect
  if steps <= 0:
    # perform diffusion until no more nodes can be activated
    return _diffuse_all(DG, A)
  # perform diffusion for at most "steps" rounds
  return _diffuse_k_rounds(DG, A, steps)

def _diffuse_all(G, A):
  tried_edges = set()
  layer_i_nodes = [ ]
  layer_i_nodes.append([i for i in A])  # prevent side effect
  while True:
    len_old = len(A)
    (A, activated_nodes_of_this_round, cur_tried_edges) = \
        _diffuse_one_round(G, A, tried_edges)
    layer_i_nodes.append(activated_nodes_of_this_round)
    tried_edges = tried_edges.union(cur_tried_edges)
    if len(A) == len_old:
      break
  return layer_i_nodes

def _diffuse_k_rounds(G, A, steps):
  tried_edges = set()
  layer_i_nodes = [ ]
  layer_i_nodes.append([i for i in A])
  while steps > 0 and len(A) < len(G):
    len_old = len(A)
    (A, activated_nodes_of_this_round, cur_tried_edges) = \
        _diffuse_one_round(G, A, tried_edges)
    layer_i_nodes.append(activated_nodes_of_this_round)
    tried_edges = tried_edges.union(cur_tried_edges)
    if len(A) == len_old:
      break
    steps -= 1
  return layer_i_nodes

def _diffuse_one_round(G, A, tried_edges):
  activated_nodes_of_this_round = set()
  cur_tried_edges = set()
  for s in A:
    for nb in G.successors(s):
      if nb in A or (s, nb) in tried_edges or (s, nb) in cur_tried_edges:
        continue
      if _prop_success(G, s, nb):
        activated_nodes_of_this_round.add(nb)
      cur_tried_edges.add((s, nb))
  activated_nodes_of_this_round = list(activated_nodes_of_this_round)
  A.extend(activated_nodes_of_this_round)
  return A, activated_nodes_of_this_round, cur_tried_edges

def _prop_success(G, src, dest):
  return random.random() <= G[src][dest]['act_prob']



# def vectorized_IC(A, nodes, seed):
#
#
#   infected_status = np.zeros(len(nodes))
#
#   # status list
#   UNINFECTED, INFECTED, REMOVED = 0, 1, 2
#
#   infected_status[seed] = INFECTED
#   infected_nodes = [seed.copy()]
#
#   while True:
#
#     current_active = np.where(infected_status == INFECTED)[0]
#     infected_nodes.append(current_active)
#
#     active_prob = A[current_active].toarray()
#
#     # print(active_prob)
#
#     next_layer = np.random.binomial(1, active_prob)
#
#     # print(next_layer)
#
#     # next_infected = np.where((next_layer == 1).any(axis=0))[0]
#
#     next_infected = next_layer.sum(0) > 0  # boolean array. True if they are infected
#
#     # print(next_infected)
#
#     next_infected = next_infected * (infected_status != REMOVED)
#
#     # print(next_infected)
#
#     # next_infected = next_infected[np.where(infected_status==2)]
#
#     infected_status[current_active] = REMOVED
#
#     infected_status[next_infected] = INFECTED
#
#     # print(infected_status)
#
#     if next_infected.sum() == 0:
#       break
#
#
#
#   return sum(infected_status == REMOVED), infected_nodes


def vectorized_IC(A, seeds):
  infected_status = np.zeros(A.shape[0])

  # status list

  UNINFECTED, ACTIVE, INFECTED = 0, 1, 2

  infected_status[seeds] = INFECTED

  infected_nodes = [seeds.copy()]

  current_active = seeds

  while len(current_active) != 0:
    active_prob = A[current_active].toarray()

    # print(f'active_prob: {active_prob}')

    # find uninfected neighbors

    neighbors = (active_prob.sum(0) > 0) * (infected_status == UNINFECTED)

    # print(f'neighbors: {neighbors}')

    next_layer = np.random.binomial(1, active_prob)

    # print(f'next_layer: {next_layer}')

    # next_infected = np.where((next_layer == 1).any(axis=0))[0]

    next_infected = (next_layer.sum(0) > 0) * neighbors  # boolean array. True if they are infected

    infected_status[current_active] = INFECTED  # indexing using list of indices

    infected_status[next_infected] = ACTIVE  # indexing using boolean array

    # update current active

    current_active = np.where(infected_status == ACTIVE)[0]

    infected_nodes.append(list(current_active))

    # print(infected_status)

  return None, infected_nodes



def evaluate_set(seed_set):

  K = 10
  n_mc = 200

  # load the graph
  with open('../../k_submodular/influence_maximization/diggs/diggs.pkl', 'rb') as f:
    G = pickle.load(f)


  # get K different adjacency matrices
  As = []
  for k in range(K):

    for u, v in G.edges:
      G[u][v]['weight'] = G[u][v][f'k_{k}']

    As.append(nx.adjacency_matrix(G.copy(), nodelist=sorted(G.nodes)))

  # setup infected nodes
  infected_nodes = {mc_id: [] for mc_id in range(n_mc)}

  for k in range(K):
    seed_k = [node_id for k_, node_id in seed_set if k == k_ ]
    print(seed_k)
    for mc_id in range(n_mc):
      _, layers = vectorized_IC(As[k], list(set(seed_k)))
      infected_nodes[mc_id].extend([j for layer in layers for j in layer])

  # merge infected nodes across the different topics
  print(infected_nodes)
  infected_nodes = np.mean([len(set(lst)) for lst in list(infected_nodes.values())])


  print(f'#Infected nodes {infected_nodes}')


def evaluate_set_old_ic(seed_set):

  K = 10
  n_mc = 200

  # load the graph
  with open('../../k_submodular/influence_maximization/diggs/diggs.pkl', 'rb') as f:
    G = pickle.load(f)


  # get K different adjacency matrices
  Gs = []
  for k in range(K):

    for u, v in G.edges:
      G[u][v]['act_prob'] = G[u][v][f'k_{k}']

    Gs.append(G.copy())

  # setup infected nodes
  infected_nodes = {mc_id: [] for mc_id in range(n_mc)}

  for k in range(K):
    seed_k = [node_id for k_, node_id in seed_set if k == k_ ]
    print(seed_k)
    for mc_id in range(n_mc):
      layers = independent_cascade(Gs[k], list(set(seed_k)))
      infected_nodes[mc_id].extend([j for layer in layers for j in layer])

  # merge infected nodes across the different topics
  print(infected_nodes)
  infected_nodes = np.mean([len(set(lst)) for lst in list(infected_nodes.values())])


  print(f'#Infected nodes {infected_nodes}')


if __name__ == '__main__':
  evaluate_set_old_ic(
    [(4, 683), (8, 551), (3, 336), (3, 142), (4, 1782), (1, 432), (2, 793), (0, 486), (2, 555), (8, 1267),
     (2, 257), (8, 966), (2, 99), (2, 83), (0, 431), (7, 635), (3, 1029), (6, 489), (6, 443),
     (5, 3087), (5, 377), (5, 171), (1, 728), (9, 899), (0, 1020), (0, 113), (8, 685), (8, 646),
     (8, 73), (7, 3157)])
  evaluate_set(
    [(4, 683), (8, 551), (3, 336), (3, 142), (4, 1782), (1, 432), (2, 793), (0, 486), (2, 555), (8, 1267), (2, 257),
     (8, 966), (2, 99), (2, 83), (0, 431), (7, 635), (3, 1029), (6, 489), (6, 443), (5, 3087), (5, 377), (5, 171),
     (1, 728), (9, 899), (0, 1020), (0, 113), (8, 685), (8, 646), (8, 73), (7, 3157)])


