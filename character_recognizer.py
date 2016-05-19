## character_recognizer.py
## Aidan Holloway-Bidwell and Jack Lightbody
## Computational Models of Cognition: Final Project

import numpy
from Node import *
from Character import *

def load_chars(filename):
    f = open(filename, 'r')
    chars = []

    for line in f:
        line = line.rstrip()
        data_list = line.split('\t')
        newChar = Character(data_list)
        chars.append(newChar)
    
    return chars

## Creates input and output list tuple from Character list
def create_training_data(chars):
    inp = []
    out = []
    for char in chars:
        out.append(char.charVal)
        inp.append(char.pixels)
    out = create_output_matrix(out)
    return inp, out

def create_output_matrix(outputs):
    matrix = numpy.zeros((len(outputs), 26))
    for i in range(len(outputs)):
        desired = ord(outputs[i]) - 97
        matrix[i][desired] = 1
    return matrix

def logistic(x):
    return 1 / (1 + numpy.exp(-x))

def dlogistic(x):
    return x * (1 - x)

def sum_squared_error(target, actual):
    return (target - actual) ** 2

def di(weight, target, inp, derivative_fn):
    return (target - inp) * derivative_fn(inp)

def multinomial_output(lst):
    run_sum = 0
    results = []
    for item in lst:
        run_sum += numpy.exp(item)
    for item in lst:
        part_sum = run_sum - np.exp(item)
        results.append(np.exp(item)/part_sum)
    return results

def delta(eta, weight, target, inp, error_fn, derivative_fn):
    return eta * error_fn(target, weight * inp) * derivative_fn(weight * inp) * inp

def learn(inputs, targets, iterations, hidden, eta):
    weights0 = numpy.random.randn(len(inputs), hidden)
    weights1 = numpy.random.randn(hidden, len(numpy.unique(targets)))
    input_nodes = []
    hidden_nodes = []
    output_nodes = []

    for inp in inputs[0]:
        input_nodes.append(Node(0))
    for hid in range(hidden):
        hidden_nodes.append(Node(0, input_nodes, weights0))
    for hid in range(len(numpy.unique(targets))):
        output_nodes.append(Node(0, hidden_nodes, weights1))

    for i in range(iterations):
        for j in range(len(inputs)):
            ## propagate inputs all the way to the output layer
            for k in range(len(input_nodes)):
                input_nodes[k].activation = inputs[k]
            for hidden in hidden_nodes:
                hidden.activate()
            for output in output_nodes:
                output.activate()
            ## back-propagate and adjust weights
            for k in range(len(output_nodes)):
                for l in range(len(output_nodes[k].inputs)):
                    weight = output_nodes[k].weights[l]
                    inp = output_nodes[k].inputs[l].activation
                    weight += delta(eta, weight, targets[k], inp, sum_squared_error, dlogistic)
            for k in range(len(hidden_nodes)):
                sum_di = 0
                for l in range(len(output_nodes)):
                    weight = output_nodes[l].weights[k]
                    sum_di += weight * di(weight, targets[l], hidden[k].activation, dlogistic)
                for l in range(len(hidden_nodes[k].inputs)):
                    weight = hidden_nodes[k].weights[l]
                    inp = hidden[k].inputs[l]
                    dj = sum_di * dlogistic(inp)
                    weight += eta * dj * inp
            ## reset activation values
            [0 for hidden.activation in hidden_nodes]
            [0 for output.activation in output_nodes]
            
    return input_nodes, hidden_nodes, output_nodes
            
def predict(inputs, input_nodes, hidden_nodes, output_nodes, hidden_fn, output_fn):
    for i in range(len(input_nodes)):
        input_nodes[i].activation = inputs[i]
    for hidden in hidden_nodes:
        hidden.activate()
    for output in output_nodes:
        output.activate()
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    