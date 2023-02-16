import os, sys;
import pickle

sys.path.append(os.path.dirname('../'))
import argparse

from multiprocessing import Pool
import numpy as np
import networkx
import pandas as pd
import matplotlib.pyplot as plt

import ohsaka
import threshold_algorithm

import weighted_network
import independent_cascade


plt.rcParams['figure.figsize'] = [10,8]
plt.rc('font', size = 30)
plt.rc('legend', fontsize = 20)


def prepare_network(file):
    ## Read network as nx.Graph
    network = networkx.read_edgelist(file, nodetype=int)

    active_nodes = []

    for n in network.nodes:
        d = network.degree(n)
        if d >= 200:
            active_nodes.append(d)

    # to directed
    network = network.to_directed()
    return network, active_nodes[:3] # TODO: REMOVE THIS


# TODO CHECK randomness is consistent
def create_K_networks(network, K):
    import random
    random.seed(1000)


    K_networks = [network.copy() for i in range(K)]
    directed_network = network.copy()

    for u, v in network.edges:

        # generate probs
        probs = [(i + 1) / (K * network.in_degree(v)) for i in range(K)]
        random.shuffle(probs)

        # weight graphs
        for i in range(K):
            K_networks[i][u][v]['act_prob'] = probs[i]

    print(len(K_networks))
    return K_networks


class Experiment:
    def __init__(self,
                 B_total,
                 B_i,
                 topics,  # topic ids to spread,
                 tolerance=None,
                 file='../../notebooks/facebook_ego.txt',
                 n_mc=30,
                 n_mc_final=10_000,
                 algorithm=ohsaka.KGreedyTotalSizeConstrained,
                 n_jobs=5

                 ):

        assert len(topics) == len(B_i), "#topics should be equal to the items to be selected"

        self.topics = topics # item id
        self.network, self.active_nodes = prepare_network(file)

        self._initialize_weighted_networks()

        self.n = len(self.active_nodes)
        print(f'Using {self.n} active users ')
        self.B_total = B_total # total budget
        self.B_i = B_i

        self.tolerance = tolerance
        self.n_mc = n_mc
        self.n_mc_final = n_mc_final
        self.n_jobs = n_jobs

        print(f'Using {self.n_jobs} jobs, n_mc {self.n_mc}')


        # initialize algorithm
        if self.tolerance is not None:
            self.algorithms = [
                algorithm(self.n,
                    self.B_total,
                    self.B_i,
                    self.value_function,tolerance=t) for t in self.tolerance]
        else:
            self.algorithms = [algorithm(self.n,
                self.B_total,
                self.B_i,
                self.value_function)]




    def _initialize_weighted_networks(self):
        # load facebook network
        self.K_networks = create_K_networks(self.network, len(self.topics))



    def value_function(self, seed_set, n_mc=None):
        n_mc = n_mc or self.n_mc
        infected_nodes = []
        print(seed_set)
        for topic_idx, topic in enumerate(self.topics):

            # filter list of users by topic(item)
            # Translate the values
            seed_t = [self.active_nodes[location_idx] for item_idx, location_idx in seed_set if item_idx == topic_idx]

            if seed_t:
                global ic_runner
                def ic_runner(t):
                    layers = independent_cascade.independent_cascade(self.K_networks[topic_idx], list(set(seed_t)))
                    infected_nodes_ = [i for l in layers for i in l]
                    return infected_nodes_

                with Pool(self.n_jobs) as p:
                    nodes = p.map(ic_runner, range(n_mc))
                    nodes = [j for i in nodes for j in i]  # flatten
                    infected_nodes.extend(nodes)


        infected_nodes = len(set(infected_nodes))
        # Influences
        print(f'Infected nodes {infected_nodes}')

        return infected_nodes


    def run(self):
        for alg in self.algorithms:
            alg.run()
    @property
    def results(self):
        return [{
            'alg': alg.name,
            'B_total': alg.B_total,
            'B_i': alg.B_i,
            'n_evals': alg.n_evaluations,
            'function_value': alg.current_value,
            'S': alg.S,
            'tolerance': self.tolerance[i] if self.tolerance is not None else None
        } for i, alg in enumerate(self.algorithms)]




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Experiment runner')
    parser.add_argument('--mode', action='store', type=str, default='run', choices=['run', 'plot'])
    parser.add_argument('--B', action='store', type=int, default=[ 2, 3, 4 ], nargs='+')
    parser.add_argument('--n-jobs', action='store', type=int, default=10)
    parser.add_argument('--tolerance', action='store', type=float, default=[0.1, 0.2], nargs='+') # TODO; update this
    parser.add_argument('--output', action='store', type=str, required=False)
    parser.add_argument('--alg', action='store', type=str, default=None,
                        choices=['KGreedyTotalSizeConstrained', 'KStochasticGreedyTotalSizeConstrained', 'ThresholdGreedyTotalSizeConstrained'])

    args = parser.parse_args()

    mode = args.mode
    B_totals = args.B
    n_jobs = args.n_jobs
    tolerance_vals = args.tolerance

    # prepare directories
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    alg_mappings = {
        'KGreedyTotalSizeConstrained': [ohsaka.KGreedyTotalSizeConstrained],
        'KStochasticGreedyTotalSizeConstrained': [ohsaka.KStochasticGreedyTotalSizeConstrained],
        'ThresholdGreedyTotalSizeConstrained': [threshold_algorithm.ThresholdGreedyTotalSizeConstrained]
    }

    algorithms = [
        ohsaka.KGreedyTotalSizeConstrained,
        ohsaka.KStochasticGreedyTotalSizeConstrained,
        threshold_algorithm.ThresholdGreedyTotalSizeConstrained
    ]

    if args.alg:
        algorithms = alg_mappings[args.alg]

    if mode == 'run':
        for alg in algorithms:
            for B_total in B_totals:
                print(f'Running experiment for {alg} with budget {B_total}')
                topics = list(range(1, 6))
                print(f'Using topics {topics}')

                exp = Experiment(
                    B_total=B_total,
                    B_i=[1] * len(topics),
                    topics=topics,
                    algorithm=alg,
                    tolerance= tolerance_vals if 'Threshold' in alg.__name__ else None,
                    n_jobs=n_jobs
                )


                exp.run()

                # save file
                with open(f'{output_dir}/{alg.__name__}__{B_total}_.pkl', 'wb') as f:
                    pickle.dump(exp.results, f)

    elif mode == 'plot':
        # load the files
        function_values = {}
        n_evaluations = {}

        algs = []

        for alg in algorithms:
            if 'Threshold' in alg.__name__:
                for t in tolerance_vals:
                    name = alg.name + f'($\epsilon$={t})'
                    function_values[name] = []
                    n_evaluations[name] = []
                    algs.append(alg)
            else:
                function_values[alg.name] = []
                n_evaluations[alg.name] = []
                algs.append(alg)

            for B_total in B_totals:
                with open(f'{output_dir}/{alg.__name__}__{B_total}_.pkl', 'rb') as f:
                    results = pickle.load(f)

                if type(results) == dict: results = [results]

                for i, r in enumerate(results):
                    if 'Threshold' in alg.__name__:
                        name = alg.name + f'($\epsilon$={tolerance_vals[i]})'
                        function_values[name].append(r['function_value'])
                        n_evaluations[name].append(r['n_evals'])
                    else:

                        function_values[alg.name].append(r['function_value'])
                        n_evaluations[alg.name].append(r['n_evals'])


        marker_types = ['o', 'v', '*', 'D', 's']
        for i, key in enumerate(function_values.keys()):
            plt.plot(range(len(B_totals)), function_values[key], label=key, marker=marker_types[i])
            plt.ylabel('Influence spread')
            plt.xlabel('Budget (B)')
            plt.xticks(range(len(B_totals)), B_totals)

        plt.legend()

        plt.savefig(f'{output_dir}/figure-infected-nodes.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.figure()

        for i, key in enumerate(function_values.keys()):
            plt.plot(range(len(B_totals)), n_evaluations[key], label=key, marker=marker_types[i])
            plt.ylabel('function evaluations')
            plt.xticks(range(len(B_totals)), B_totals)
            plt.xlabel('Total Size (b)')
        plt.legend()

        plt.savefig(f'{output_dir}/figure-function-evaluations.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.figure()
