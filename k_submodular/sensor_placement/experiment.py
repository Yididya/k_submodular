import os, sys;
import pickle



sys.path.append(os.path.dirname('../'))
import argparse

import numpy as np
import matplotlib.pyplot as plt

from k_submodular import ohsaka
from k_submodular import threshold_algorithm
import pandas as pd


plt.rcParams['figure.figsize'] = [10,8]
plt.rc('font', size = 30)
plt.rc('legend', fontsize = 20)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)


with open('./sensor_data.pkl', 'rb') as f:
    sensor_data = pickle.load(f)



class Experiment:
    def __init__(self,
                 B_total,
                 B_i,
                 tolerance=None, # tolerance parameter(epsilon) for the threshold
                 delta=None,
                 algorithm=ohsaka.KGreedyIndividualSizeConstrained,
                 ):

        humidity = pd.read_csv('../../k_submodular/sensor_placement/hums.tsv', sep='\t')
        temp = pd.read_csv('../../k_submodular/sensor_placement/temps.tsv', sep='\t')
        light = pd.read_csv('../../k_submodular/sensor_placement/lights.tsv', sep='\t')

        # remove last column
        humidity = humidity[humidity.columns[:55]]
        temp = temp[temp.columns[:55]]
        light = light[light.columns[:55]]

        # sensor locations
        self.location_ids = humidity.columns.to_list()  # location ids
        self.sensor_types = ['temp', 'humidity', 'light']

        # rename and merge
        light.rename(columns={k: f'light_{k}' for k in light.columns}, inplace=True)
        temp.rename(columns={k: f'temp_{k}' for k in temp.columns}, inplace=True)
        humidity.rename(columns={k: f'humidity_{k}' for k in humidity.columns}, inplace=True)

        self.df = pd.concat([temp, humidity, light], axis=1)

        self.location_maps = dict(zip(range(len(self.location_ids)), self.location_ids))
        self.sensor_maps = dict(zip(range(len(self.sensor_types)), self.sensor_types))

        self.n = len(self.location_ids)
        self.B_total = B_total  # total budget
        self.B_i = B_i

        self.k = len(B_i)
        assert self.k == 3, "Only three readings considered"

        self.tolerance = tolerance
        self.delta = delta

        # initialize algorithm
        if self.tolerance is not None:
            self.algorithms = [
                algorithm(self.n,
                    self.B_total,
                    self.B_i.copy(),
                    self.value_function,
                    epsilon=t) for t in self.tolerance]
        elif self.delta is not None:
            self.algorithms = [algorithm(self.n,
                self.B_total,
                self.B_i,
                self.value_function, delta=d) for d in self.delta]
        else:
            self.algorithms = [algorithm(self.n,
                                         self.B_total,
                                         self.B_i,
                                         self.value_function)]

    def calculate_entropy(self, cols):
        n_data_points = self.df.shape[0]

        freqs = self.df.groupby(cols).size().values

        p = freqs / n_data_points

        H = - np.sum(p * np.log(p))

        return H

    def value_function(self, seed_set):
        cols = []
        for sensor_idx, location_idx in seed_set:
            cols.append(f'{self.sensor_types[sensor_idx]}_{self.location_ids[location_idx]}')

        return self.calculate_entropy(cols)


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
            'tolerance': self.tolerance[i] if self.tolerance is not None else None,
            'delta': self.delta[i] if self.delta is not None else None
        } for i, alg in enumerate(self.algorithms)]


def dict_to_file(dict_, file):
    table = list(dict_.values())
    df = pd.DataFrame(table)
    df = df.transpose()
    df.columns = list(dict_.keys())
    df.to_csv(file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Experiment runner')
    parser.add_argument('--mode', action='store', type=str, default='plot', choices=['run', 'plot'])
    parser.add_argument('--B', action='store', type=int, default=None, nargs='+')
    parser.add_argument('--B_i', action='store', type=int, default=list(range(1, 19)), nargs='+')
    parser.add_argument('--tolerance', action='store', type=float, default=[0.1, 0.2, 0.5, 0.8], nargs='+')
    parser.add_argument('--delta', action='store', type=float, default=[0.1, 0.2, 0.5, 0.8], nargs='+')

    parser.add_argument('--output', action='store', type=str, required=False)
    parser.add_argument('--alg', action='store', type=str, default=None,
                        choices=[
                            'KGreedyIndividualSizeConstrained',
                            'KStochasticGreedyIndividualSizeConstrained',
                            'ThresholdGreedyIndividualSizeConstrained'
                        ])

    args = parser.parse_args()

    mode = args.mode
    B_totals = args.B
    tolerance_vals = args.tolerance
    delta_vals = args.delta
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
                    tolerance= tolerance_vals if 'Threshold' in alg.__name__ else None,
                    delta= delta_vals if 'KGreedy' in alg.__name__ or 'KStochastic' in alg.__name__ else None
                )


                exp.run()

                # save file
                if 'Threshold' in alg.__name__:
                    with open(f'{output_dir}/{alg.__name__}__{B_total}_{tolerance_vals[0]}.pkl', 'wb') as f:
                        pickle.dump(exp.results, f)
                elif 'KStochastic' in alg.__name__:
                    with open(f'{output_dir}/{alg.__name__}__{B_total}_{delta_vals[0]}.pkl', 'wb') as f:
                        pickle.dump(exp.results, f)
                else:
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
            elif 'KStochastic' in alg.__name__:
                for d in delta_vals:
                    name = alg.name + f'($\delta$={d})'
                    function_values[name] = []
                    n_evaluations[name] = []
                    algs.append(alg)
            else:
                name = alg.name
                function_values[name] = []
                n_evaluations[name] = []
                algs.append(alg)

            for B_total in B_totals:
                if 'Threshold' in alg.__name__:
                    results = []
                    for t_val in tolerance_vals:
                        with open(f'{output_dir}/{alg.__name__}__{B_total}_{t_val}.pkl', 'rb') as f:
                            results.extend(pickle.load(f))
                elif 'KStochastic' in alg.__name__:
                    results = []
                    for d in delta_vals:
                        with open(f'{output_dir}/{alg.__name__}__{B_total}_{d}.pkl', 'rb') as f:
                            results.extend(pickle.load(f))
                else:
                    with open(f'{output_dir}/{alg.__name__}__{B_total}_.pkl', 'rb') as f:
                        results = pickle.load(f)

                if type(results) == dict: results = [results]

                for i, r in enumerate(results):
                    if 'Threshold' in alg.__name__:
                        name = alg.name + f'($\epsilon$={tolerance_vals[i]})'
                        function_values[name].append(r['function_value'])
                        n_evaluations[name].append(r['n_evals'])
                    elif 'KStochastic' in alg.__name__:
                        name = alg.name + f'($\delta$={delta_vals[i]})'

                        function_values[name].append(r['function_value'])
                        n_evaluations[name].append(r['n_evals'])
                    else:
                        name = alg.name

                        function_values[name].append(r['function_value'])
                        n_evaluations[name].append(r['n_evals'])


        marker_types = ['o', 'v', '*', 'D', 'x', 'H', 1, 8, 9, 'D', 6]
        for i, key in enumerate(function_values.keys()):
            plt.plot(range(len(B_totals)), function_values[key], label=key, marker=marker_types[i])
            plt.ylabel('Entropy')
            plt.xlabel('Value of b')
            plt.xticks(range(len(B_totals)), [int(b / 3) for b in B_totals])

        plt.legend()

        plt.savefig(f'{output_dir}/figure-entropy.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.figure()

        for i, key in enumerate(function_values.keys()):
            plt.plot(range(len(B_totals)), n_evaluations[key], label=key, marker=marker_types[i])
            plt.ylabel('Function Evaluations')
            plt.xticks(range(len(B_totals)), [int(b / 3) for b in B_totals])
            plt.xlabel('Value of b')
        plt.legend()

        plt.savefig(f'{output_dir}/figure-function-evaluations.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.figure()

        # save to file
        dict_to_file(function_values, f'{output_dir}/function_values.csv')
        dict_to_file(n_evaluations, f'{output_dir}/n_evaluations.csv')


