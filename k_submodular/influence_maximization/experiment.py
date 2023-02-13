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



class FriendDataset:

    def __init__(self, file):
        self.data = pd.read_csv(file, sep='\t')

    @property
    def users(self):
        return list(self.data['user_id'].unique())

    def user_friends(self, user_id):
        return self.data[self.data['user_id'] == user_id ]['friend_id'].unique()




class Experiment:
    def __init__(self,
                 B_total,
                 B_i,
                 topics,  # topic ids to spread,
                 tolerance=None,
                 action_log_file='../../notebooks/output/digg_filtered_friends.txt',
                 friend_list_file='../../notebooks/output/digg_filtered_friends.txt',
                 weights_file='../../notebooks/output/digg_with_weights.txt_InfProbs(10)',
                 n_mc=30,
                 n_mc_final=10_000,
                 algorithm=ohsaka.KGreedyTotalSizeConstrained,
                 n_jobs=5

                 ):

        assert len(topics) == len(B_i), "#topics should be equal to the items to be selected"

        self.topics = topics # item id
        self.action_log_file = action_log_file
        self.friends_dataset = FriendDataset(friend_list_file)

        self.user_ids = self.friends_dataset.users
        print(f'Using {len(self.user_ids)} users')


        self.n = len(self.user_ids)
        self.B_total = B_total # total budget
        self.B_i = B_i

        self.tolerance = tolerance
        self.n_mc = n_mc
        self.n_mc_final = n_mc_final
        self.n_jobs = n_jobs

        print(f'Using {self.n_jobs} jobs, n_mc {self.n_mc}')


        self.networks = {}
        self.weights = {}

        self._load_weights(weights_file)
        self._initialize_weighted_networks()
        print(f'Using K={len(self.networks.keys())} ')

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

        G = networkx.DiGraph()
        users = self.friends_dataset.users

        for u in users:
            G.add_node(u)
            friends = self.friends_dataset.user_friends(u)

            G.add_edges_from([(u, f) for f in friends])

        # Add weight to the network
        for topic_id in self.topics:
            G_topic = G.copy()
            self.networks[topic_id] = weighted_network.weighted_network(G_topic, None, weights=self.weights[topic_id])
            # self.networks[topic_id] = weighted_network.weighted_network(G_topic, 'rn')

    def _load_weights(self, file):
        data = pd.read_csv(file, sep='\t', names=['Influencer', 'Influenced'] + [f'Topic{t}' for t in self.topics], header=0, index_col=False)

        for topic_id in self.topics:
            topic_col = f'Topic{topic_id}'
            sliced_data = data[['Influencer', 'Influenced', f'Topic{topic_id}']].to_dict('list')

            influencer, influenced, weights_ = sliced_data['Influencer'], sliced_data['Influenced'], sliced_data[topic_col]

            self.weights[topic_id]  = {(influencer[i], influenced[i]): weights_[i] for i, _ in enumerate(weights_)}

    def value_function(self, seed_set, n_mc=None):
        n_mc = n_mc or self.n_mc
        infected_nodes = []
        for topic_idx, topic in enumerate(self.topics):

            # filter list of users by topic(item)
            # Translate the values
            seed_t = [self.user_ids[location_idx] for item_idx, location_idx in seed_set if item_idx == topic_idx]

            if seed_t:
                # TODO: if this is the up
                global ic_runner
                def ic_runner(t):
                    layers = independent_cascade.independent_cascade(self.networks[topic], list(set(seed_t)))
                    infected_nodes_ = [i for l in layers for i in l]
                    return infected_nodes_

                with Pool(self.n_jobs) as p:
                    nodes = p.map(ic_runner, range(n_mc))
                    nodes = [j for i in nodes for j in i]  # flatten
                    infected_nodes.extend(nodes)

        # Influences
        return len(set(infected_nodes))


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
    parser.add_argument('--B', action='store', type=int, default=[2, 3], nargs='+')
    parser.add_argument('--n-jobs', action='store', type=int, default=20)
    parser.add_argument('--tolerance', action='store', type=float, default=[0.1, 0.2], nargs='+')
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
                topics = list(range(1, 11))

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


        marker_types = ['o', 'v', '*', 'D']
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
