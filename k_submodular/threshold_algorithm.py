import numpy as np
import ohsaka




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

        max_gain = 0.
        # for each available positions
        V_avail = self._V_available.copy()
        for v in V_avail:
            # calculate the gain of placing item i on it
            for i in range(self.K): # over K item types 
                gain = self.marginal_gain(i, v, update_count=False)
                if gain > max_gain:
                    max_gain = gain 
        return max_gain 


    def _calculate_min_threshold(self):
        return self.tolerance * self._d * (1 - self.tolerance) / (2 * self.B_total)




    @property
    def U_S(self):
        """
        set of locations(indices) that are already occupied/filled by some item i
        """
        return [v for i, v in self.S]



    def run(self):
        ## no real rounds

        budget_exhausted = False  

        while self.threshold > self._min_threshold:

            V_avail = self._V_available.copy()
            for v in V_avail:
                for i in range(self.K):
                    # TODO: assume one location only allows one item  -- len(S) == len(U(S))
                    budget_exhausted = len(self.S) == self.B_total 
                    
                    if not budget_exhausted:
                        # check table, make sure the saved marginal gain is >= current threshold
                        # if yes, evaluate and update
                        lookup_value = self.lookup_marginal(i, v)
                        if lookup_value < self.threshold:
                            # don't bother
                            continue
                        gain = self.marginal_gain(i, v)
                        if gain >= self.threshold:
                            # add (item, index) pair to list and 
                            self._V_available.remove(v)
                            self.V[v] = i
                            self.S.append((i, v))
                            self.current_value += gain
                            print(f'{self.__class__.__name__} - Added element idx {v}/ {self.B_total}')
                            break

                    else:
                        break
                        
            if budget_exhausted:
                break
                    
            # update threshold 
            self.threshold = (1 - self.tolerance) * self.threshold

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


    def run(self):
        
        while self.threshold > self._min_threshold:

            V_avail = self._V_available.copy()
            for v in V_avail:
                B_i_remaining = self.B_i_remaining.copy()
                for i, available in enumerate(B_i_remaining): # over K item types
                    if available != 0:
                        lookup_value = self.lookup_marginal(i, v)
                        if lookup_value < self.threshold:
                            # don't bother
                            continue
                        gain = self.marginal_gain(i, v)
                        if gain >= self.threshold:
                            # add (item, index) pair to list and 
                            self._V_available.remove(v)
                            self.V[v] = i
                            self.S.append((i, v))
                            self.current_value += gain
                            
                            # Decrement the value of B_i
                            self.B_i_remaining[i] -= 1
                            print(f'{self.__class__.__name__} - Added element idx {v}/ {self.B_total}')
                            break

           
                        
          
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
    B_total = 5

    value_function = value_function_template(n, B_i)

    experiment = ThresholdGreedyTotalSizeConstrained(n, B_total=B_total, B_i = B_i, value_function=value_function, tolerance=0.4)
    # experiment = ThresholdGreedyIndividualSizeConstrained(n, B_total=B_total, B_i = B_i, value_function=value_function, tolerance=0.4)

    experiment.run()
    print(experiment)