## character_recognizer.py
## Aidan Holloway-Bidwell and Jack Lightbody
## Computational Models of Cognition: Final Project
from __future__ import division
import numpy
from Node import *
import random
import time
def load_chars(filename):
    f = open(filename, 'r')
    chars = []
    matrix = []
    lists = []
    inp = []
    nextlists = []
    for line in f:
        line = line.rstrip()
        data_list = line.split('\t')
        lists.append(data_list)
#    random.shuffle(lists)
#    for data_list in lists:
        pixels = list(map(int, data_list[6:]))
        pixels = compute_pixel_sums(pixels)
        inp.append(pixels)
        charVal = data_list[1]
        desired = ord(charVal) - 97
        arr = numpy.zeros(26)
        arr[desired] = 1
        matrix.append(arr)
    ##compute_next_prev(lists, matrix)
    return inp, matrix

##def compute_next_prv(lists, output_matrix)
##    prev = []
##    nexts = []
##    inps = []
##    for i in range(len(lists)):
##        inps.append(numpy.concatenate(prev, nexts)
##        del prev[0]
##        prev.append(output_matrix[i])

def compute_pixel_sums(pixels):
    pixels = numpy.reshape(pixels, (16, 8))
    matrix = numpy.zeros((8, 8))
    for i in range(len(matrix)):
        for k in range(len(matrix[0])):
            matrix[i][k] = (pixels[2*i][k]+pixels[(2*i)+1][k])/2
    return numpy.reshape(matrix, 64)

## Creates input and output list tuple from Character list
def create_training_data(chars):
    inp = []
    out = []
    for char in chars:
        out.append(char.charVal)
        inp.append(char.pixels)
    out = create_output_matrix(out)
    return inp, out

def get_confusion_matrix(output, desired):
    matrix = numpy.zeros((26, 26)) # make the empty 10*10 matrix
    i = 0
    right = 0
    for item in output:
        desired_val = numpy.argmax(desired[i])
        real_val = numpy.argmax(item)
        matrix[desired_val][real_val]+=1 # fill as we go
        if desired_val == real_val:
            right+=1
        i+=1
    print right
    print i
    return matrix

def create_output_matrix(outputs):
    matrix = numpy.zeros((len(outputs), 26))
    for i in range(len(outputs)):
        desired = ord(outputs[i]) - 97
        matrix[i][desired] = 1
    return matrix

def logistic(x):
    return 1 / (1 + numpy.exp(-x))

def dlogistic(x):
    return x* (1 - x)

def sum_squared_error(target, actual):
    return (target - actual) ** 2
def dsum_squared_error(target, actual):
    return (target - actual)

def di(weight, target, inp, error_fn, derivative_fn):
    return error_fn(target, inp*weight) * derivative_fn(inp*weight)

def multinomial_output(lst):
    run_sum = 0
    results = []
    for item in lst:
        run_sum += numpy.exp(item.activation)
    for item in lst:
        part_sum = run_sum - numpy.exp(item.activation)
        results.append(numpy.exp(item.activation)/part_sum)
    return results

def delta(eta, weight, target, inp, error_fn, derivative_fn):
    return eta * error_fn(target, weight * inp) * derivative_fn(weight * inp) * inp

def tan(x):
    return (numpy.exp(x)-numpy.exp(-x))/(numpy.exp(x)+numpy.exp(-x))

def dtan(x):
    return 1-((numpy.exp(x)-numpy.exp(-x))**2/(numpy.exp(x)+numpy.exp(-x))**2)

def learn2(inputs, targets, iterations, hid, eta):
    inputs = numpy.asarray(inputs).T

    weights0 = numpy.random.randn(inputs.shape[0], hid)
    weights1 = numpy.random.randn(26, hid)

    for i in range(iterations):
        for inp in inputs:
            hidden = numpy.dot(weights0, inp.T)
            print hidden

    print(hidden)


def learn(inputs, targets, iterations, hidden, eta):
    weights0 = numpy.random.randn(hidden, len(inputs[0]))
    weights1 = numpy.random.randn(26, hidden)
    input_nodes = []
    hidden_nodes = []
    output_nodes = []
    actTime = 0
    hidback = 0
    outback = 0
    for inp in inputs[0]:
        input_nodes.append(Node(0))
    for hid in range(hidden):
        hidden_nodes.append(Node(0, input_nodes, weights0[hid]))
    for target in range(26):
        output_nodes.append(Node(0, hidden_nodes, weights1[target]))
    for i in range(iterations):
        print i
        for j in range(len(inputs)):
            start_time = time.time()
            ## propagate inputs all the way to the output layer
            for k in range(len(inputs[j])):
                input_nodes[k].activation = inputs[j][k]
            for hidden in hidden_nodes:
                hidden.activate(tan)
            for output in output_nodes:
                output.activate(tan)
            actTime+=time.time()-start_time
            ## back-propagate and adjust weights
            start_time = time.time()
            for k in range(len(output_nodes)):
                for l in range(len(output_nodes[k].inputs)):
                    weight = output_nodes[k].weights[l]
                    inp = output_nodes[k].inputs[l].activation
                    deltav = delta(eta, weight, targets[j][k], inp, dsum_squared_error, dtan)
                    output_nodes[k].weights[l] += deltav
                output_nodes[k].activation = 0
            outback+=time.time()-start_time
            start_time = time.time()
            for k in range(len(hidden_nodes)):
                sum_di = 0
                for l in range(len(output_nodes)):
                    weight = output_nodes[l].weights[k]
                    sum_di += weight * di(weight, targets[j][l], hidden_nodes[k].activation, dsum_squared_error, dtan)
                for l in range(len(hidden_nodes[k].inputs)):
                    weight = hidden_nodes[k].weights[l]
                    inp = hidden_nodes[k].inputs[l]
                    dj = sum_di * dtan(inp.activation*weight)
                    change = eta * dj * inp.activation
                    hidden_nodes[k].weights[l] += change
                hidden_nodes[k].activation = 0
            hidback+=time.time()-start_time
    print actTime
    print hidback
    print outback
    return input_nodes, hidden_nodes, output_nodes

def predictOutputs(inputs, input_nodes, hidden_nodes, output_nodes, output_fn):
    results = []
    for inputVal in inputs:
        out = predict(inputVal, input_nodes, hidden_nodes, output_nodes)
        results.append(output_fn(out))
    return results

def predict(inputVal, input_nodes, hidden_nodes, output_nodes):
    for i in range(len(input_nodes)):
        input_nodes[i].activation = inputVal[i]
    for hidden in hidden_nodes:
        hidden.activation = 0
        hidden.activate(tan)
    for output in output_nodes:
        output.activation = 0
        output.activate(tan)
    return output_nodes
