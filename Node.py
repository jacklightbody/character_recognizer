class Node:
    inputs = []
    weights = []
    activation = 0

    def logistic(x):
        return 1 / (1 + numpy.exp(-x))
    
    def activate(self, activation_fn = logistic):
        print self.inputs
        for i in range(len(self.inputs)):
            self.activation += activation_fn(self.inputs[i].activation * self.weights[i])

    def __init__(self, base, inp=None, wts=None):
        self.activation = base
        self.inputs = inp
        self.weights = wts