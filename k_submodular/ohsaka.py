
from this import d
import numpy as np
from abc import abstractmethod
import random 
import heapq as hq
import functools

def value_function_template(n, B_i):
    # generate a lookup table 
    lookup_table = np.random.random(size=(len(B_i), n))
    
    
    # Add predefined values to check sanity 
    lookup_table[0][0] = 9.
    lookup_table[0][1] = 5.

    print(lookup_table)

    def value_function(item_index_pairs):
        total = 0.
        for i, v in item_index_pairs:
            total += lookup_table[i][v]

        return total

    return value_function


@functools.total_ordering
class ItemIndexPair(object):
    def __init__(self, pair_index, marginal_gain=None, last_evaluated=0):
        self.index = pair_index
        self.marginal_gain = marginal_gain or -np.inf
        self.last_evaluated = last_evaluated

    def __lt__(self, other):
        """
        Priority comparision,
        the less the better, thus __lt__ instead of  __gt__
        """
        return self.marginal_gain > other.marginal_gain

    def __eq__(self, other):
        return (self.last_evaluated, self.marginal_gain) == (other.last_evaluated, other.marginal_gain)

    def __repr__(self):
        return '(ItemIndexPair {}, Last Evaluated {}, Marginal Gain {} )'.format(self.index,
                                                                                 self.last_evaluated,
                                                                                 self.marginal_gain)


class KSubmodular():

    def __init__(self, 
                n, 
                B_total, 
                B_i, 
                value_function):
        """
       
        :param n - number of locations(indices)
        :param B_total - int - Total budget constraint 
        :param B_i  - budget constraint on items of type i -- list of budgets.  |B_i| is the total types of items 
        :param V - total number of locations("indices")
        :param value_function - takes (set of item-index pair)
            item-index pair examples
                Sensor application - item - sensor type , index - the index of the location (out of "n" locations ) where the sensor is placed
                Influence maximization item - the k kinds of items to distribute(free-stuff), the index of person out of n people to which the item is assigned (Ohsaka et. al)

        """
        self.n = n
        self.B_i = B_i
        self.B_total = B_total

        self.V = [-1 for v in range(self.n) ] # universe of locations(indices)
        self._V_available = [i  for i, _ in enumerate(self.V)] # filtered available indices

        self.S = [] # item-index pairs currently selected  -- (i, v) 

        self.value_function = value_function

        # current value of the function 
        self.current_value = 0.

        self.n_evaluations = 0

        ## marginal_gain lookup table for lazy evaluation
        self.marginal_lookup_table = np.ones((self.K, self.n)) * np.inf

    def pair_pool(self, V_available=None):
        pool = []

        V_avail = V_available or self._V_available

        # initialize heap with available elements
        for i in range(self.K):
            for v in V_avail:
                pool.append(ItemIndexPair((i, v), marginal_gain=self.marginal_lookup_table[i][v]))

        hq.heapify(pool)
        return pool

    def lookup_marginal(self, i, v):
        return self.marginal_lookup_table[i][v]


    def update_marginal(self, i, v, value):
        self.marginal_lookup_table[i][v] = max(value, 0.)


    @property
    def K(self):
        return len(self.B_i)    



    def marginal_gain(self, i, v, update_count=True):
        """
        marginal gain of adding item i onto index v
        :param i - item i, on index v
        """
        assert self.V[v] == -1, 'void already filled'
        if update_count:
            self.n_evaluations += 1

        value = self.value_function(self.S + [(i, v)]) - self.current_value

        if update_count:
            self.update_marginal(i, v, value)

        return value


    def support(self):
        return [v for idx, v in enumerate(self.V) if v != -1]

    @abstractmethod
    def run(self): 
        pass



class KGreedyTotalSizeConstrained(KSubmodular):
    name = 'k-Greedy-TS'

    def __init__(self, 
                n, 
                B_total, 
                B_i, 
                value_function):
 
        super().__init__(
            n,
            B_total,
            B_i,
            value_function
        )
        

    def run(self):
        
        # until the budget is exhausted 
        for j in range(self.B_total):
            print(f'{self.__class__.__name__} - Iteration {j}/{self.B_total}')
            max_item, max_value = (None, None), -np.inf
            pool = self.pair_pool()

            for _ in range(len(pool)):
                # get an element out of the loop
                item = hq.heappop(pool)
                i, v = item.index
                lookup_value = self.lookup_marginal(i, v)
                if lookup_value < max_value:
                    # don't bother
                    continue
                gain = self.marginal_gain(i, v)
                if gain > max_value:
                    max_item, max_value = (i, v), gain

                
            # update V_available 
            if max_item[0] is not None and max_item[1] is not None:
                print(f'Selected {max_item} with gain {max_value}')
                self._V_available.remove(max_item[1])
                self.V[max_item[1]] = max_item[0]
                self.S.append(max_item)

                self.current_value += max_value
        assert len(self.S) == self.B_total, "Budget must be used up"
        print(self.S)
        print(f'Final value {self.current_value}')
        



class KStochasticGreedyTotalSizeConstrained(KSubmodular):
    name = 'k-Stochastic-Greedy-TS'

    def __init__(self, 
                n, 
                B_total, 
                B_i, 
                value_function,
                delta=0.1):
 
        super().__init__(
            n,
            B_total,
            B_i,
            value_function
        )

        self.delta = delta
        print(f'Using delta -- {self.delta}')



    def run(self):

        # until the budget is exhausted
        for j in range(self.B_total):
            print(f'{self.__class__.__name__} - Iteration {j}/{self.B_total}')
            max_item, max_value = (None, None), -np.inf

            # compute random V_available
            V_avail = self._V_available.copy()
            subset_size = min(
                int((self.n - j + 1) / (self.B_total - j + 1) * np.log(self.B_total / self.delta)),
                self.n
            )
            # Random sample
            if subset_size < len(V_avail):
                V_avail = random.sample(V_avail, k=subset_size)

            pool = self.pair_pool(V_available=V_avail) # restrict to the pool only to the random choices

            for _ in range(len(pool)):
                # get an element out of the loop
                item = hq.heappop(pool)
                i, v = item.index
                lookup_value = self.lookup_marginal(i, v)
                if lookup_value < max_value:
                    # don't bother
                    continue
                gain = self.marginal_gain(i, v)
                if gain > max_value:
                    max_item, max_value = (i, v), gain

            # update V_available
            if max_item[0] is not None and max_item[1] is not None:
                print(f'Selected {max_item} with gain {max_value}')
                self._V_available.remove(max_item[1])
                self.V[max_item[1]] = max_item[0]
                self.S.append(max_item)

                self.current_value += max_value

        assert len(self.S) == self.B_total, "Budget must be used up"
        print(self.S)
        print(f'Final value {self.current_value}')







class KGreedyIndividualSizeConstrained(KSubmodular):
    name = 'k-Greedy-IS'

    def __init__(self, 
                n, 
                B_total, 
                B_i, 
                value_function):
        

        super().__init__(
            n,
            B_total,
            B_i,
            value_function
        )

        self.B_total = sum(self.B_i)
        self.B_i_remaining = self.B_i.copy()

    def pair_pool(self, V_available=None):
        pool = []

        V_avail = V_available or self._V_available

        # initialize heap with available elements
        for i, available in enumerate(self.B_i_remaining):
            if available > 0:
                for v in V_avail:
                    pool.append(ItemIndexPair((i, v), marginal_gain=self.marginal_lookup_table[i][v]))

        hq.heapify(pool)
        return pool

    def run(self):
        
        # until the budget is exhausted 
        for j in range(self.B_total):
            print(f'{self.__class__.name} - Iteration {j}/{self.B_total}')
            max_item, max_value = (None, None), -np.inf
            V_avail = self._V_available.copy()


            pool = self.pair_pool(V_available=V_avail)  # restrict to the pool only to the random choices

            for _ in range(len(pool)):
                # get an element out of the loop
                item = hq.heappop(pool)
                i, v = item.index
                lookup_value = self.lookup_marginal(i, v)
                if lookup_value < max_value:
                    # don't bother
                    continue
                gain = self.marginal_gain(i, v)
                if gain > max_value:
                    max_item, max_value = (i, v), gain

                
            # update V_available 
            if max_item[0] is not None:
                print(f'Selected {max_item} with gain {max_value}')
                self._V_available.remove(max_item[1])
                self.V[max_item[1]] = max_item[0]
                self.S.append(max_item)
                self.current_value += max_value

                # Decrement the value of B_i
                self.B_i_remaining[max_item[0]] -= 1
        assert len(self.S) == self.B_total, "Budget must be used up"
        print(self.S)
        print(f'Final value {self.current_value}')



class KStochasticGreedyIndividualSizeConstrained(KGreedyIndividualSizeConstrained):
    name = 'k-Greedy-Stochastic-IS'
    
    def __init__(self, 
            n, 
            B_total, 
            B_i, 
            value_function,
            delta=0.1):

        super().__init__(n, B_total, B_i, value_function)

        self.delta = delta
        print(f'Using delta -- {self.delta}')
        self.B_total = sum(self.B_i)
        self.B_i_remaining = self.B_i.copy()


    def support_i(self, i):
        """
        List of supported indices(locations) by items of type i 
        """

        return [ v for idx, v in self.S if idx == i]
    
    def subset_size_i(self, i):
        """
        Size of random subset appropriate for i
        """

        ith_support = len(self.support_i(i))

        return min (
            int((self.n - ith_support) / (self.B_i[i] - ith_support) * np.log(self.B_total / self.delta)),
            self.n
        )



    def run(self):
        # until the budget is exhausted 
        for j in range(self.B_total):
            print(f'{self.__class__.__name__} - Iteration {j}/{self.B_total}')
            R = []

            max_item, max_value = (None, None), -np.inf

            while True: 
                # add a single element to R
                V_avail = self._V_available.copy()
                choices = [v for v in V_avail if v not in R]
                if choices:
                    choice = random.choice(choices)
                    R.append(choice)

                    v = R[-1]
                    for i, available in enumerate(self.B_i_remaining): # over K item types
                        if available != 0:
                            lookup_value = self.lookup_marginal(i, v)
                            if lookup_value < max_value:
                                # don't bother
                                continue
                            gain = self.marginal_gain(i, v)
                            if gain > max_value:
                                max_item, max_value = (i, v), gain

                if max_item[0] is not None and len(R) >= self.subset_size_i(max_item[0]) or not choices:
                    # update V_available
                    print(f'Selected {max_item} with gain {max_value}')
                    self._V_available.remove(max_item[1])
                    self.V[max_item[1]] = max_item[0]
                    self.S.append(max_item)
                    self.current_value += max_value

                    # Decrement the value of B_i
                    self.B_i_remaining[max_item[0]] -= 1

                    break

        assert len(self.S) == self.B_total, "Budget must be used up"
        print(self.S)
        print(f'Final value {self.current_value}')

if __name__ == '__main__':
    n = 10 
    B_i = [2, 3, 4]
    B_total = sum(B_i)

    value_function = value_function_template(n, B_i)
    experiment = KGreedyTotalSizeConstrained(n, B_total, B_i=B_i, value_function=value_function)


    experiment.run()
    print(f'Number of evaluations {experiment.n_evaluations}')
    assert experiment.n_evaluations == (n * len(B_i) + B_total - 1), 'Lazy evaluation sanity check'

    # total size greedy
    experiment = KStochasticGreedyTotalSizeConstrained(n, B_total=B_total, B_i = B_i, value_function=value_function)

    experiment.run()
    assert experiment.n_evaluations <= (n * len(B_i) + B_total - 1), 'Lazy evaluation sanity check'
    print(f'Number of evaluations {experiment.n_evaluations }')


    # individual greedy
    experiment = KGreedyIndividualSizeConstrained(n, B_total=B_total, B_i=B_i, value_function=value_function)
    experiment.run()
    for i, b_i in enumerate(B_i):
        assert len([s for s in experiment.S if s[0] == i]) == b_i
    print(f'Number of evaluations {experiment.n_evaluations}')
    assert experiment.n_evaluations <= (n * len(B_i) + B_total - 1), 'Lazy evaluation sanity check'
    assert len(experiment._V_available) + len(experiment.S) == len(experiment.V)
    # # individual greedy
    # experiment = KStochasticGreedyIndividualSizeConstrained(n, B_total=B_total, B_i=B_i, value_function=value_function)
    # experiment.run()
    # print(f'Number of evaluations {experiment.n_evaluations}')