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


plt.rcParams['figure.figsize'] = [10,8]
plt.rc('font', size = 30)
plt.rc('legend', fontsize = 20)


with open('./sensor_data.pkl', 'rb') as f:
    sensor_data = pickle.load(f)



class Experiment:
    def __init__(self,
                 B_total,
                 B_i,
                 tolerance=None,
                 n_mc=30,
                 n_mc_final=10_000,
                 algorithm=ohsaka.KGreedyIndividualSizeConstrained,
                 ):




        # sensor locations
        self.location_ids = sensor_data['locations']  # location ids
        self.location_maps = dict(zip(range(len(self.location_ids)), self.location_ids))

        self.entropy_data = sensor_data['temperature_entropies'], \
                            sensor_data['humidity_entropies'], \
                            sensor_data['light_entropies']

        self.n = len(self.location_ids)
        self.B_total = B_total # total budget
        self.B_i = B_i

        self.k = len(B_i)
        assert self.k == 3, "Only three readings considered"

        self.tolerance = tolerance

        # initialize algorithm
        if self.tolerance is not None:
            self.algorithms = [
                algorithm(self.n,
                    self.B_total,
                    self.B_i,
                    self.value_function,
                    tolerance=t) for t in self.tolerance]
        else:
            self.algorithms = [algorithm(self.n,
                self.B_total,
                self.B_i,
                self.value_function)]





    def value_function(self, seed_set):
        seed_set = [self.location_maps[location_idx] for s, location_idx in seed_set]

        total_entropy = 0.
        for i, entropy in enumerate(self.entropy_data):
            total_entropy += sum([entropy[int(s)] for s in seed_set])


        # Influences
        return total_entropy


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
    parser.add_argument('--B', action='store', type=int, default=None, nargs='+')
    parser.add_argument('--B_i', action='store', type=int, default=list(range(1, 11)), nargs='+')
    parser.add_argument('--tolerance', action='store', type=float, default=[0.1, 0.2], nargs='+')
    parser.add_argument('--output', action='store', type=str, required=False)
    parser.add_argument('--alg', action='store', type=str, default=None,
                        choices=['KGreedyIndividualSizeConstrained', 'KStochasticGreedyIndividualSizeConstrained', 'ThresholdGreedyIndividualSizeConstrained'])

    args = parser.parse_args()

    mode = args.mode
    B_totals = args.B
    tolerance_vals = args.tolerance
    K = 3
    B_i_s = [[B_i] * K for B_i in args.B_i ]
    B_totals = [ sum(B_i) for B_i in B_i_s ]

    # prepare directories
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    alg_mappings = {
        'KGreedyIndividualSizeConstrained': [ohsaka.KGreedyIndividualSizeConstrained],
        'KStochasticGreedyIndividualSizeConstrained': [ohsaka.KStochasticGreedyIndividualSizeConstrained],
        'ThresholdGreedyIndividualSizeConstrained': [threshold_algorithm.ThresholdGreedyIndividualSizeConstrained]
    }

    algorithms = [
        ohsaka.KGreedyIndividualSizeConstrained,
        ohsaka.KStochasticGreedyIndividualSizeConstrained,
        threshold_algorithm.ThresholdGreedyIndividualSizeConstrained
    ]

    if args.alg:
        algorithms = alg_mappings[args.alg]

    if mode == 'run':
        for alg in algorithms:
            for j, B_total in enumerate(B_totals):
                print(f'Running experiment for {alg} with budget {B_total}')

                exp = Experiment(
                    B_total=B_total,
                    B_i=B_i_s[j].copy(),
                    algorithm=alg,
                    tolerance= tolerance_vals if 'Threshold' in alg.__name__ else None
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
            plt.ylabel('Entropy')
            plt.xlabel('Budget (B)')
            plt.xticks(range(len(B_totals)), B_totals)

        plt.legend()

        plt.savefig(f'{output_dir}/figure-entropy.png', dpi=300, bbox_inches='tight')
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
