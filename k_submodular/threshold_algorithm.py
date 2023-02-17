import numpy as np
import ohsaka
import heapq as hq




class ThresholdGreedyTotalSizeConstrained(ohsaka.KSubmodular):
    name = "Threshold-Greedy-TS"

    def __init__(self, 
        n, 
        B_total, 
        B_i, 
        value_function,
        tolerance):
        """
        :param tolerance(epsilon) parameter with which the threshold(tau) is defined 
        """

        super().__init__(
            n,
            B_total,
            B_i,
            value_function
        )

        self.tolerance = tolerance
        # self._d = self._calculate_d()
        # self._min_threshold = self._calculate_min_threshold()
        # self.threshold = self._d

    @property
    def tolerance(self):
        return self._tolerance


    @tolerance.setter
    def tolerance(self, val):
        self._tolerance = val
        self._d = self._calculate_d()
        self._min_threshold = self._calculate_min_threshold()
        self.threshold = self._d

    def _calculate_d(self):
        """
        Finding the maximum value of the function   
        """

        max_item, max_gain = (None, None), 0.
        # for each available positions
        V_avail = self._V_available.copy()
        for v in V_avail:
            # calculate the gain of placing item i on it
            for i in range(self.K): # over K item types 
                gain = self.marginal_gain(i, v)
                if gain > max_gain:
                    max_gain = gain
                    max_item = (i, v)

        # add the element to the initial set S
        if max_item[0] is not None:
            self._V_available.remove(max_item[1])
            self.V[max_item[1]] = max_item[0]
            self.S.append(max_item)
            self.current_value += max_gain

        return max_gain 


    def _calculate_min_threshold(self):
        return self.tolerance * self._d * (1 - self.tolerance) / (2 * self.B_total)





    def run(self):
        ## no real rounds

        budget_exhausted = False  

        while self.threshold > self._min_threshold:

            pool = self.pair_pool()

            for _ in range(len(pool)):
                # get an element out of the loop
                item = hq.heappop(pool)
                i, v = item.index
                budget_exhausted = len(self.S) == self.B_total

                if not budget_exhausted and self.V[v] == -1:  # budget available & item is still empty
                    # check table, make sure the saved marginal gain is >= current threshold
                    # if yes, evaluate and update
                    lookup_value = self.lookup_marginal(i, v)
                    if lookup_value < self.threshold:
                        # don't bother
                        break
                    gain = self.marginal_gain(i, v)
                    if gain >= self.threshold:
                        # add (item, index) pair to list and
                        self._V_available.remove(v)
                        self.V[v] = i
                        self.S.append((i, v))
                        self.current_value += gain
                        print(f'{self.__class__.__name__} - {len(self.S) } / {self.B_total}')

            if budget_exhausted:
                break
                    
            # update threshold 
            self.threshold = (1 - self.tolerance) * self.threshold

        # pad remaining values
        pool = self.pair_pool()
        remaining_count = self.B_total - len(self.S)
        while remaining_count > 0 and len(pool) > 0:
            # get an element out of the loop
            # check availability of v
            item = hq.heappop(pool)
            i, v = item.index

            if self.V[v] == -1:
                print('Padding more items ')
                self.V[v] = i
                self.S.append(item.index)
                remaining_count -= 1




        print(self.S)
        print(f'Final value {self.current_value}')
        
        return 





class ThresholdGreedyIndividualSizeConstrained(ThresholdGreedyTotalSizeConstrained):
    name = "Threshold-Greedy-IS"

    def __init__(self, 
        n, 
        B_total, 
        B_i, 
        value_function,
        tolerance):
        """
        :param tolerance(epsilon) parameter with which the threshold(tau) is defined 
        """

        super().__init__(
            n,
            B_total,
            B_i,
            value_function,
            tolerance=tolerance
        )
        
        self._min_threshold = self._calculate_min_threshold()
        print(f'Using initial min_threshold {self._min_threshold} with tolerance {self.tolerance}')
        self.B_i_remaining = self.B_i.copy()

    @property
    def budget_exhausted(self):
        return sum(self.B_i_remaining) == 0


    
    def support_i(self, i):
        """
        List of supported indices(locations) by items of type i 
        """

        return [ v for idx, v in self.S if idx == i]


    def _calculate_min_threshold(self):
        return self.tolerance * self._d * (1 - self.tolerance) / (3 * self.B_total)


    def pair_pool(self, V_available=None):
        pool = []

        V_avail = V_available or self._V_available

        # initialize heap with available elements
        for i, available in enumerate(self.B_i_remaining):
            if available > 0:
                for v in V_avail:
                    pool.append(ohsaka.ItemIndexPair((i, v), marginal_gain=self.marginal_lookup_table[i][v]))

        hq.heapify(pool)
        return pool

    def run(self):
        
        while self.threshold > self._min_threshold:

            pool = self.pair_pool()

            for _ in range(len(pool)):
                # get an element out of the loop
                item = hq.heappop(pool)
                i, v = item.index
                budget_exhausted = len(self.S) == self.B_total

                if not budget_exhausted and self.V[v] == -1:  # budget available & item is still empty
                    # check table, make sure the saved marginal gain is >= current threshold
                    # if yes, evaluate and update
                    lookup_value = self.lookup_marginal(i, v)
                    if lookup_value < self.threshold:
                        # don't bother
                        break
                    gain = self.marginal_gain(i, v)
                    if gain >= self.threshold:
                        # add (item, index) pair to list and
                        self._V_available.remove(v)
                        self.V[v] = i
                        self.S.append((i, v))
                        self.current_value += gain

                        # Decrement the value of B_i
                        self.B_i_remaining[i] -= 1
                        print(f'{self.__class__.__name__} - {len(self.S) } / {self.B_total}')


          
            # update threshold 
            self.threshold = (1 - self.tolerance) * self.threshold

            # check on budget 
            if self.budget_exhausted:
                break
                
        
        print(self.S)
        print(f'Final value {self.current_value}')

        return




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



if __name__ == '__main__':
    n = 10 
    B_i = [2, 3, 4]
    B_total = sum(B_i)

    value_function = value_function_template(n, B_i)

    experiment = ThresholdGreedyTotalSizeConstrained(n, B_total=B_total, B_i = B_i, value_function=value_function, tolerance=0.1)
    experiment.run()
    print(f'Number of evaluations {experiment.n_evaluations}')

    # individual size
    experiment = ThresholdGreedyIndividualSizeConstrained(n, B_total=B_total, B_i = B_i, value_function=value_function, tolerance=0.4)
    experiment.run()
    print(f'Number of evaluations {experiment.n_evaluations}')