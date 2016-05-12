## character_recognizer.py
## Aidan Holloway-Bidwell and Jack Lightbody
## Computational Models of Cognition: Final Project

import numpy

class Character:
    charVal = None
    pixels = []

    def __init__(self, data_list):
        charVal = data_list[1]
        pixels = list(map(int, data_list[6:]))

class Node:
    inputs = []
    weights = []
    activation = 0

    def activate(self, activation_fn = logistic):
        for i in range(len(inputs)):
            activation += activation_fn(inputs[i] * weights[i])

    def __init__(self, base, inp=None, wts=None):
        activation = base
        inputs = inp
        weights = wts

def logistic(x):
    return 1 / (1 + numpy.exp(-x))

def load_chars(filename):
    f = open(filename, 'r')
    chars = []

    for line in f:
        line = line.rstrip()
        data_list = line.split('\t')
        newChar = Character(data_list)
        chars.append(newChar)

    return chars

def multinomial_output(lst):
    run_sum = 0
    results = []
    for item in lst:
        run_sum += numpy.exp(item)
    for item in lst:
        part_sum = run_sum - np.exp(item)
        results.append(np.exp(item)/part_sum)
    return results

## Creates input and output list tuple from Character list
def create_training_data(chars):
    inp = []
    out = []
    for char in chars:
        inp.append(char.charVal)
        out.append(char.pixels)

    out = create_output_matrix(out)
    return inp, out

def create_output_matrix(outputs):
    matrix = numpy.zeros(len(outputs), len(numpy.unique(outputs)))
    for i in range(len(outputs)):
        desired = ord(outputs[i])- 61
        matrix[i][desired] = 1
    return matrix

def learn(inputs, outputs, iterations, hidden, eta):
    weights0 = numpy.random.randn(len(inputs), hidden)
    weights1 = numpy.random.randn(hidden, len(numpy.unique(outputs)))
    input_nodes = []
    hidden_nodes = []
    output_nodes = []

    for inp in inputs[0]:
        input_nodes.append(Node(0))
    for hid in range(hidden):
        hidden_nodes.append(Node(0, input_nodes, weights0))
    for hid in range(len(numpy.unique(outputs))):
        output_nodes.append(Node(0, hidden_nodes, weights1))

    for i range(iterations):
        for j in range(len(inputs)):
            for k in range(len(input_nodes)):
                input_nodes[k].activation = inp[k]
            for hidden in hidden_nodes:
                hidden.activate()
            for output in output_nodes:
                output.activate()


    for i in range(iterations):
