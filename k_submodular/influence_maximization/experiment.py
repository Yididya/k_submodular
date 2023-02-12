import networkx

import pandas as pd
from ..ohsaka import KGreedyTotalSizeConstrained, KStochasticGreedyTotalSizeConstrained
from ..threshold_algorithm import ThresholdGreedyTotalSizeConstrained


class FriendDataset:

    def __init__(self, file):
        self.data = pd.read_csv(file)

    @property
    def users(self):
        return self.data['user_id'].to_list()

    def user_friends(self, user_id):
        return self.data[self.data['user_id'] == user_id ]['friend_id'].unique()




class Experiment:


    def __init__(self,
                 n,
                 B_total,
                 B_i,
                 topics=[], # topic ids to spread
                 action_log_file='../../notebooks/output/digg_filtered_friends.txt',
                 friend_list_file='../../notebooks/output/digg_filtered_friends.txt',
                 weights_file='../../notebook/output/digg_with_weights.txt_InfProbs(10)',
                 n_mc=30,
                 n_mc_final=10_000,
                 algorithm=KGreedyTotalSizeConstrained,

                 ):

        self.topics = topics # item id
        self.action_log_file = action_log_file
        self.friends_dataset = FriendDataset(friend_list_file)


        self.n = n
        self.B_total = B_total # total budget
        self.B_i = B_i
        self.algorithm = algorithm

        # initialize algorithm
        algorithm(self.n,
            self.B_total,
            self.B_i,
            self.value_function)

        self.n_mc = n_mc
        self.n_mc_final = n_mc_final


        self.networks = []
        self._initialize_weighted_networks()



    def _initialize_weighted_network(self):

        G = networkx.DiGraph()
        users = self.friends_dataset.users

        for u in users:
            G.add_node(u)
            friends = self.friends_dataset.user_friends(u)

            G.add_edges_from([(u, f) for f in friends])


        for topic_id in self.topics:
            G.copy()




    def value_function(self, seed, n_times):
        pass



