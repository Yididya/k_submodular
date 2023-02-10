from .algorithm import KSubmodular




class ThresholdGreedyTotalSizeConstrained(KSubmodular):


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
        self.d = self._calculate_d()


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


    def min_threshold(self):
        return (1 - self.tolerance) / (2 * self.B_total)




    @property
    def U_S(self):
        """
        set of locations(indices) that are already occupied/filled by some item i
        """
        return [v for i, v in self.S]



    def run(self):
        ## no real rounds 

        return 
