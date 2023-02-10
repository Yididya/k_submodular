import numpy as np
from algorithm import KSubmodular




class ThresholdGreedyTotalSizeConstrained(KSubmodular):
    name = "k-submodular-Greedy-TS"

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

        self._tolerance = tolerance
        self._d = self._calculate_d()
        self._min_threshold = self._calculate_min_threshold()
        self.threshold = self._d 



    def _calculate_d(self):
        """
        Finding the maximum value of the function   
        """

        max_gain = 0.
        # for each available positions 
        for v in self.V_available():
            # calculate the gain of placing item i on it
            for i in range(self.K): # over K item types 
                gain = self.marginal_gain(i, v)
                if gain > max_gain:
                    max_gain = gain 
        return max_gain 


    def _calculate_min_threshold(self):
        return self._d * (1 - self._tolerance) / (2 * self.B_total)




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
            

            for v in self.V_available():
                for i in range(self.K):
                    # TODO: assume one location only allows one item  -- len(S) == len(U(S))
                    budget_exhausted = len(self.S) == self.B_total 
                    
                    if not budget_exhausted:
                        gain = self.marginal_gain(i, v)
                        
                        if gain >= self.threshold:
                            # add (item, index) pair to list and 
                            self._V_available.remove(v)
                            self.S.append((i, v))
                            self.current_value += gain

                    else:
                        break
                        
            if budget_exhausted:
                break
                    
            # update threshold 
            self.threshold = (1 - self._tolerance) * self.threshold

        print(self.S)
        print(f'Final value {self.current_value}')
        
        return 





class ThresholdGreedyIndividualSizeConstrained(ThresholdGreedyTotalSizeConstrained):
    name = "k-submodular-Greedy-IS"

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


    @property
    def budget_exhausted(self):
        return sum(self.B_i)


    
    def support_i(self, i):
        """
        List of supported indices(locations) by items of type i 
        """

        return [ idx for idx, v in self.S if idx == i]


    def _calculate_min_threshold(self):
        return self._d * (1 - self._tolerance) / (3 * self.B_total)


    def run(self):
        
        while self.threshold > self._min_threshold:
            
            for v in self.V_available():
                for i, available in enumerate(self.B_i): # over K item types 
                    if available != 0:
                        gain = self.marginal_gain(i, v)
                        if gain >= self.threshold:
                            # add (item, index) pair to list and 
                            self._V_available.remove(v)
                            self.S.append((i, v))
                            self.current_value += gain
                            
                            # Decrement the value of B_i
                            self.B_i[i] -= 1

           
                        
          
            # update threshold 
            self.threshold = (1 - self._tolerance) * self.threshold

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

    experiment = ThresholdGreedyIndividualSizeConstrained(n, B_total=B_total, B_i = B_i, value_function=value_function, tolerance=0.4)

    experiment.run()
    print(experiment)