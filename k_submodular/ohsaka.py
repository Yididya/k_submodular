
from this import d
import numpy as np
from abc import abstractmethod
import random 


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

        self.V = [-1 for v in range(n) ] # universe of locations(indices)
        self._V_available = [i  for i, _ in enumerate(self.V)] # filtered available indices 

        self.S = [] # item-index pairs currently selected  -- (i, v) 

        self.value_function = value_function

        # current value of the function 
        self.current_value = 0.

        self.n_evaluations = 0


    @property
    def K(self):
        return len(self.B_i)    


    def V_available(self):
        return self._V_available


    def marginal_gain(self, i, v):
        """
        marginal gain of adding item i onto index v
        :param i - item i, on index v
        """
        assert self.V[v] == -1, 'void already filled'
        self.n_evaluations += 1

        return self.value_function(self.S + [(i, v)]) - self.current_value


    def support(self):
        return [idx for idx, v in enumerate(self.V) if v != -1] 

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
            max_item, max_value, gain = (None, None), 0., 0.
    
            for v in self.V_available():
                for i in range(self.K): # over K item types 
                    gain = self.marginal_gain(i, v)
                    if gain > max_value:
                        max_item, max_value = (i, v), gain
                
            # update V_available 
            if max_item[0] is not None:
                self._V_available.remove(max_item[1])
                self.S.append(max_item)
                self.current_value += gain

        assert len(self.S) == self.B_total, "Budget must be used up" 
        
        print(self.S)
        print(f'Final value {self.current_value}')
        



class KStochasticGreedyTotalSizeConstrained(KGreedyTotalSizeConstrained):
    name = 'k-Greedy-Stochastic-TS'

    def __init__(self, 
                n, 
                B_total, 
                B_i, 
                value_function,
                delta=0.5):
 
        super().__init__(
            n,
            B_total,
            B_i,
            value_function
        )

        self.delta = delta


    @property
    def subset_size(self):
        """
        todo log base 
        """
        j = len(self.S) + 1 
        return min(
            int((self.n - j + 1)/ (self.B_total - j + 1 ) * np.log(self.B_total / self.delta)),
            self.n
        )

    def V_available(self):
        return random.choices(self._V_available, k=self.subset_size)
    


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


    def run(self):
        
        # until the budget is exhausted 
        for j in range(self.B_total):
            max_item, max_value, gain = (None, None), 0., 0.
    
            for v in self.V_available():
                for i, available in enumerate(self.B_i): # over K item types 
                    if available != 0:
                        gain = self.marginal_gain(i, v)
                        if gain > max_value:
                            max_item, max_value = (i, v), gain
                
            # update V_available 
            if max_item[0] is not None:
                self._V_available.remove(max_item[1])
                self.S.append(max_item)
                self.current_value += gain

                # Decrement the value of B_i
                self.B_i[max_item[0]] -= 1

        assert len(self.S) == self.B_total, "Budget must be used up" 
        
        print(self.S)
        print(f'Final value {self.current_value}')



class KStochasticGreedyIndividualSizeConstrained(KGreedyIndividualSizeConstrained):
    name = 'k-Stochastic-Greedy-IS'
    
    def __init__(self, 
            n, 
            B_total, 
            B_i, 
            value_function,
            delta=0.5):

        super().__init__(n, B_total, B_i, value_function)
        self.delta = delta 

    def support_i(self, i):
        """
        List of supported indices(locations) by items of type i 
        """

        return [ idx for idx, v in self.S if idx == i]
    
    def subset_size_i(self, i):
        """
        Size of random subset appropriate for i
        """

        ith_support = len(self.support_i(i))

        return min (
            int((self.n - ith_support) / (self.B_i[i] - len(self.support_i)) * np.log(self.B_total / self.delta)),
            n
        )



    def run(self):
        # until the budget is exhausted 
        for j in range(self.B_total):
            R = []

            while True: 
                # add a single element to R
                choice = random.choice([v for v in self.V_available() if v not in R])
                R.append(choice)


                max_item, max_value, gain = [], (None, None), 0., 0.

                # Find the maximum 
                for v in R:
                    for i, available in enumerate(self.B_i): # over K item types 
                        if available != 0:
                            gain = self.marginal_gain(i, v)
                            if gain > max_value:
                                max_item, max_value = (i, v), gain
                
                if max_item[0] is not None and len(R) >= self.subset_size_i(max_item[0]):
                    # update V_available 
                    self._V_available.remove(max_item[1])
                    self.S.append(max_item)
                    self.current_value += gain

                    # Decrement the value of B_i
                    self.B_i[max_item[0]] -= 1

                    break
                

        assert len(self.S) == self.B_total, "Budget must be used up" 
        
        print(self.S)
        print(f'Final value {self.current_value}')

if __name__ == '__main__':
    n = 10 
    B_i = [2, 3, 4]
    B_total = 5

    value_function = value_function_template(n, B_i)

    # experiment = KStochasticGreedyTotalSizeConstrained(n, B_total=B_total, B_i = B_i, value_function=value_function)
    experiment = KGreedyIndividualSizeConstrained(n, B_total=B_total, B_i = B_i, value_function=value_function)

    experiment.run()
    print(experiment)