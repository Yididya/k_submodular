import hashlib
import os, sys
import pickle
import time

from database import Database

sys.path.append(os.path.dirname('../'))
import argparse

from multiprocessing import Pool
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

import ohsaka
import threshold_algorithm

import independent_cascade


plt.rcParams['figure.figsize'] = [10,8]
plt.rc('font', size = 30)
plt.rc('legend', fontsize = 22)

def prepare_network():
    with open('../../k_submodular/influence_maximization/diggs/diggs.pkl', 'rb') as f:
        G = pickle.load(f)

    users = list(G.nodes)


    return G, users


def create_K_networks(network, K):

    K_networks = [network.copy() for _ in range(K)]

    for u, v in network.edges:
        for i in range(K):
            # K_networks[i][u][v]['act_prob'] = K_networks[i][u][v][f'k_{i}']
            K_networks[i][u][v]['weight'] = K_networks[i][u][v][f'k_{i}']



    # to adjacency matrix
    K_networks = [ nx.adjacency_matrix(G, nodelist=sorted(G.nodes)) for G in K_networks]

    return K_networks


class Experiment:
    def __init__(self,
                 B_total,
                 B_i,
                 topics,  # topic ids to spread,
                 tolerance=None,
                 n_mc=50,
                 n_mc_final=10_000,
                 algorithm=ohsaka.KGreedyTotalSizeConstrained,
                 n_jobs=5,
                 function_evalutions_dir='./output/evals',
                 write_db=False
                 ):

        assert len(topics) == len(B_i), "#topics should be equal to the items to be selected"

        self.topics = topics # item id
        self.network, self.active_nodes = prepare_network()

        self._initialize_weighted_networks()

        self.n = len(self.active_nodes)
        print(f'Using {self.n} active users ')
        self.B_total = B_total # total budget
        self.B_i = B_i

        self.tolerance = tolerance
        self.n_mc = n_mc
        self.n_mc_final = n_mc_final
        self.n_jobs = n_jobs

        ## saving expensive function evaluations
        self.function_evaluations_dir = function_evalutions_dir
        ## connect to database
        self.database = Database(filename=f'{function_evalutions_dir}/evals.db')
        self.write_db = write_db
        self.n_evaluations = 0


        ## create directory if not exists
        os.makedirs(self.function_evaluations_dir, exist_ok=True)

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
        self.K_networks = create_K_networks(self.network, len(self.topics))


    def hash_seed_set(self, seed_set):
        """
        Hashes a seed set using SHA256 algorithm and return its digest
        Returns hex digest
        -------
        """

        # order the seed set
        sorted_seed_set = sorted(seed_set)
        return hashlib.sha256(str(sorted_seed_set).encode('utf-8')).hexdigest()

    def lookup_value(self, key):
        # lookup in the database
        result = self.database.fetch_one(key)
        if result:
            print('Reading evaluation from database ')
            return result[1] ## the number of infected nodes

        # lookup in saved results
        fname = f'{self.function_evaluations_dir}/{key}.txt'
        if os.path.exists(fname):
            print('looking up saved evaluation from file..')
            with open(fname, 'r') as f:
                line = f.readline()
                vals = line.strip().split('|')
                n_infected = float(vals[0])

                return n_infected

        return None


    def save_value(self, seed_set, key, value):
        """

        Parameters
        ----------
        seed_set - seed_set
        key - hash nodes
        value - the number of infected nodes

        Returns
        -------

        """
        # save results to file
        fname = f'{self.function_evaluations_dir}/{key}.txt'

        with open(fname, 'w') as f:
            f.write(f'{value}|{seed_set}')



    def value_function(self, seed_set, n_mc=None):
        # increment number of evaluations
        self.n_evaluations += 1


        key = self.hash_seed_set(seed_set)
        value = self.lookup_value(key)

        if self.write_db and self.n_evaluations % 5 == 0: ## update db every 10 evaluations
            # update the db
            print('Updating database...')
            self.database.update_db(self.function_evaluations_dir)

        if value:
            return value

        n_mc = n_mc or self.n_mc
        infected_nodes = {i:[] for i in range(n_mc)}
        print(seed_set)
        for topic_idx, topic in enumerate(self.topics):

            # filter list of users by topic(item)
            # Translate the values
            seed_t = [self.active_nodes[location_idx] for item_idx, location_idx in seed_set if item_idx == topic_idx]

            if seed_t:
                start_time = time.time()
                global ic_runner
                def ic_runner(t):
                    # layers = independent_cascade.independent_cascade(self.K_networks[topic_idx], list(set(seed_t)))
                    _, layers = independent_cascade.vectorized_IC(self.K_networks[topic_idx], list(set(seed_t)))
                    infected_nodes_ = [i for l in layers for i in l]
                    return infected_nodes_

                with Pool(self.n_jobs) as p:
                    nodes = p.map(ic_runner, range(n_mc))
                    for i, n in enumerate(nodes):
                        infected_nodes[i].extend(n)

                total_time = time.time() - start_time
                print(f'Total time {total_time}')

        # Aggregate infected_nodes over MC runs
        infected_nodes = np.mean([len(set(lst)) for lst in list(infected_nodes.values())])

        self.save_value(seed_set, key, infected_nodes)



        return infected_nodes


    def run(self):
        for alg in self.algorithms:
            alg.run()


    def final_run(self, S_list, n_mc=200):
        """
        Parameters
        ----------
        S_list list of selected values
        Returns a dictionary with the evaluations of S on corresponding algorithms
        ------
        """
        assert len(S_list) == len(self.algorithms), 'Number of algorithms and seed set do not match'
        final_vals = []
        for S in S_list:
            final_vals.append(self.value_function(S, n_mc=n_mc))

        return final_vals



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
    parser.add_argument('--mode', action='store', type=str, default='run', choices=['run', 'plot', 'final'])
    # parser.add_argument('--B', action='store', type=int, default=[ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], nargs='+')
    parser.add_argument('--B', action='store', type=int, default=[1, 2, 3, 4], nargs='+')

    parser.add_argument('--n-jobs', action='store', type=int, default=10)
    parser.add_argument('--n-mc', action='store', type=int, default=None)
    parser.add_argument('--tolerance', action='store', type=float, default=[0.1, 0.2, 0.5], nargs='+') # TODO; update this
    parser.add_argument('--output', action='store', type=str, required=False)
    parser.add_argument('--write-db', action='store_true', default=False)
    parser.add_argument('--alg', action='store', type=str, default=None,
                        choices=['KGreedyTotalSizeConstrained', 'KStochasticGreedyTotalSizeConstrained', 'ThresholdGreedyTotalSizeConstrained'])

    args = parser.parse_args()

    mode = args.mode
    B_totals = args.B
    n_jobs = args.n_jobs
    tolerance_vals = args.tolerance
    n_mc = args.n_mc or 1
    n_mc_final = 500
    # topics = range(1, 5)
    topics = range(0, 10) #  ALL 10 TOPICS
    print(f'Using Tolerance vals {tolerance_vals}, n_mc {n_mc}')
    print(f'Option: Writing access - {args.write_db}')

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
                print(f'Using topics {topics}')
                exp = Experiment(
                    B_total=B_total,
                    B_i=[1] * len(topics),
                    topics=topics,
                    algorithm=alg,
                    tolerance= tolerance_vals[:1] if 'Threshold' in alg.__name__ else None,
                    n_jobs=n_jobs,
                    n_mc=n_mc,
                    write_db=args.write_db
                )

                exp.run()

                # save file
                if 'Threshold' in alg.__name__:
                    with open(f'{output_dir}/{alg.__name__}__{B_total}_{tolerance_vals[0]}.pkl', 'wb') as f:
                        pickle.dump(exp.results, f)
                else:
                    with open(f'{output_dir}/{alg.__name__}__{B_total}_.pkl', 'wb') as f:
                        pickle.dump(exp.results, f)
    elif mode == 'final':
        for alg in algorithms:
            for B_total in B_totals:
                # look at pickles
                if 'Threshold' in alg.__name__:
                    with open(f'{output_dir}/{alg.__name__}__{B_total}_{tolerance_vals[0]}.pkl', 'rb') as f:
                        results = pickle.load(f)
                else:
                    with open(f'{output_dir}/{alg.__name__}__{B_total}_.pkl', 'rb') as f:
                        results = pickle.load(f)

                # if results[0].get('final_function_value', None):
                #     print('Already calculated ')
                #     continue

                print(f'Running final run for {alg} with budget {B_total}')
                print(f'Using topics {topics}')
                exp = Experiment(
                    B_total=B_total,
                    B_i=[1] * len(topics),
                    topics=topics,
                    algorithm=alg,
                    tolerance= tolerance_vals[:1] if 'Threshold' in alg.__name__ else None,
                    n_jobs=n_jobs,
                    n_mc=n_mc,
                    write_db=args.write_db
                )

                final_vals = exp.final_run([r['S'] for r in results], n_mc=n_mc_final)
                for k, r in enumerate(results):
                    r['final_function_value'] = final_vals[k]


                # update pickles
                if 'Threshold' in alg.__name__:
                    with open(f'{output_dir}/{alg.__name__}__{B_total}_{tolerance_vals[0]}.pkl', 'wb') as f:
                        pickle.dump(results, f)
                else:
                    with open(f'{output_dir}/{alg.__name__}__{B_total}_.pkl', 'wb') as f:
                        pickle.dump(results, f)



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
                if 'Threshold' in alg.__name__:
                    results = []
                    for t_val in tolerance_vals:
                        with open(f'{output_dir}/{alg.__name__}__{B_total}_{t_val}.pkl', 'rb') as f:
                            results.extend(pickle.load(f))
                else:
                    with open(f'{output_dir}/{alg.__name__}__{B_total}_.pkl', 'rb') as f:
                        results = pickle.load(f)

                # if type(results) == dict: results = [results]

                for i, r in enumerate(results):
                    if 'Threshold' in alg.__name__:
                        name = alg.name + f'($\epsilon$={tolerance_vals[i]})'
                        function_values[name].append(r.get('final_function_value', r['function_value']))
                        n_evaluations[name].append(r['n_evals'])
                    else:

                        function_values[alg.name].append(r.get('final_function_value', r['function_value']))
                        n_evaluations[alg.name].append(r['n_evals'])


        marker_types = ['o', 'v', '*', 'D', 's']
        for i, key in enumerate(function_values.keys()):
            plt.plot(range(len(B_totals)), function_values[key], label=key, marker=marker_types[i], markersize=12)

            plt.xticks(range(len(B_totals)), B_totals)
        plt.ylabel('Influence Spread')
        plt.xlabel('Total Budget')
        plt.grid(axis='both')
        plt.legend()

        plt.savefig(f'{output_dir}/TS-influence-n51-k4.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.figure()

        for i, key in enumerate(function_values.keys()):
            plt.plot(range(len(B_totals)), n_evaluations[key], label=key, marker=marker_types[i], markersize=12)

            plt.xticks(range(len(B_totals)), B_totals)
            plt.ylabel('Function Evaluations')
            plt.xlabel('Total Budget')
        plt.grid(axis='both')
        plt.legend()

        plt.savefig(f'{output_dir}/TS-eval-n51-k4.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.figure()
