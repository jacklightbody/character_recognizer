## character_recognizer.py
## Aidan Holloway-Bidwell and Jack Lightbody
## Computational Models of Cognition: Final Project
from __future__ import division
import numpy
from Node import *
import random
import time
import string
import matplotlib.pyplot as plt
import colormaps as cmaps

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
    return inp, lists, matrix

def compute_next_prev(lists, output_matrix):
    zero = numpy.zeros(26)
    prevs = [zero, zero]
    nexts = [output_matrix[1], output_matrix[2]]
    endincontext = False
    endindex = 4
    inps = []
    for i in range(len(lists)):
        arr = numpy.reshape(numpy.asarray(prevs + nexts), 104)
        inps.append(arr)
        del prevs[0]
        if len(nexts) >0:
            del nexts[0]
        if not endincontext:
            if i >= len(output_matrix):
                return inps
            prevs.append(output_matrix[i])
            if len(output_matrix) > i+3:
                if lists[i+3][2] == -1:
                    endincontext = True
                    endindex = 3
                nexts.append(output_matrix[i+3])
            else:
                prevs = [zero, zero]
                nexts = [zero, zero]
        else:
            if endindex > 2:
                endindex -= 1
            else:
                prevs = []
                nexts = []
                if len(lists) > i+3:
                    prevs = [zero, zero]
                    nexts = [output_matrix[i+2], output_matrix[i+3]]
                    endincontext = False
                else:
                    prevs = [zero, zero]
                    nexts = [zero, zero]
                    endincontext = False
    inps = numpy.asarray(inps)
    return inps

def compute_context(lists, output_matrix):
    zero = numpy.zeros(26)
    context = [zero, zero, output_matrix[0], output_matrix[1], output_matrix[2]]
    endincontext = False
    endindex = 4
    inps = []
    for i in range(len(lists)):
        arr = numpy.reshape(numpy.asarray(context), 130)
        inps.append(arr)
        if len(context) == 5:
            del context[0]
        if not endincontext:
            if len(lists) > i+3:
                if lists[i+3][2] == -1:
                    endincontext = True
                    endindex = 3
                context.append(output_matrix[i+3])
            else:
                context = [zero, zero, zero, zero, zero]
        else:
            if endindex > 1:
                endindex -= 1
            else:
                prevs = []
                nexts = []
                if len(lists) > i+3:
                    context = [zero, zero, output_matrix[i+1], output_matrix[i+2], output_matrix[i+3]]
                    endincontext = False
                else:
                    context = [zero, zero, zero, zero, zero]
    inps = numpy.asarray(inps)
    return inps

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
        run_sum += numpy.exp(item)
    for item in lst:
        part_sum = run_sum - numpy.exp(item)
        results.append(numpy.exp(item)/part_sum)
    return results

def delta(eta, weight, target, inp, error_fn, derivative_fn):
    return eta * error_fn(target, weight * inp) * derivative_fn(weight * inp) * inp

def tan(x):
    return (numpy.exp(x)-numpy.exp(-x))/(numpy.exp(x)+numpy.exp(-x))

def dtan(x):
    return 1-((numpy.exp(x)-numpy.exp(-x))**2/(numpy.exp(x)+numpy.exp(-x))**2)

def learn2(inputs, targets, iterations, hid, eta, weights0, weights1, activ_fn=logistic, dactiv_fn=dlogistic, error_fn=dsum_squared_error):
    for i in range(iterations):
        print i
        for inp in range(inputs.shape[1]):
            ## propagate through to output layer
            in_layer = inputs[:, [inp]]
            hid_layer = activ_fn(numpy.dot(weights0, in_layer))
            out_layer = activ_fn(numpy.dot(weights1, hid_layer))

            ## adjust weights for hidden layer to output layer
            di = error_fn(targets[:, [inp]], out_layer) * dactiv_fn(out_layer)
            dw1 = eta * numpy.dot(di, hid_layer.T)
            weights1 += dw1
            dj = numpy.dot(di.T, weights1).T * dactiv_fn(hid_layer)
            dw0 = eta * numpy.dot(dj, in_layer.T)
            weights0 += dw0

    return weights0, weights1

def predict2(inputs, weights0, weights1, activ_fn=logistic):
    inputs = numpy.asarray(inputs)
    h_inp = numpy.dot(weights0, inputs.T)
    hidden = activ_fn(h_inp)
    o_inp = numpy.dot(weights1, hidden)
    outputs = activ_fn(o_inp)

    return outputs
def learn2Init(inputs, targets, iterations, hid, eta, activ_fn=logistic, dactiv_fn=dlogistic, error_fn=dsum_squared_error):
    inputs = numpy.asarray(inputs).T
    targets = numpy.asarray(targets).T
    weights0 = numpy.random.randn(hid, inputs.shape[0])
    weights1 = numpy.random.randn(targets.shape[0], hid)
    return learn2(inputs, targets, iterations, hid, eta, weights0, weights1, logistic, dlogistic, dsum_squared_error)

def learnInit(inputs, targets, iterations, hidden, eta):
    weights0 = numpy.random.randn(hidden, len(inputs[0]))
    weights1 = numpy.random.randn(26, hidden)
    input_nodes = []
    hidden_nodes = []
    output_nodes = []
    for inp in inputs[0]:
        input_nodes.append(Node(0))
    for hid in range(hidden):
        hidden_nodes.append(Node(0, input_nodes, weights0[hid]))
    for target in range(26):
        output_nodes.append(Node(0, hidden_nodes, weights1[target]))
    return learn(input_nodes, hidden_nodes, output_nodes, inputs, targets, iterations, hidden, eta)

def learn(input_nodes, hidden_nodes, output_nodes, inputs, targets, iterations, hidden, eta):
    for i in range(iterations):
        print i
        for j in range(len(inputs)):
            ## propagate inputs all the way to the output layer
            for k in range(len(inputs[j])):
                input_nodes[k].activation = inputs[j][k]
            for hidden in hidden_nodes:
                hidden.activate(tan)
            for output in output_nodes:
                output.activate(tan)
            ## back-propagate and adjust weights
            for k in range(len(output_nodes)):
                for l in range(len(output_nodes[k].inputs)):
                    weight = output_nodes[k].weights[l]
                    inp = output_nodes[k].inputs[l].activation
                    deltav = delta(eta, weight, targets[j][k], inp, sum_squared_error, dtan)
                    output_nodes[k].weights[l] += deltav
                output_nodes[k].activation = 0
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
    return input_nodes, hidden_nodes, output_nodes

def predictOutputs(inputs, weights0, weights1, output_fn):
    results = []
    for inputVal in inputs:
        out = predict2(inputVal, weights0, weights1)
        results.append(output_fn(out))
    return results

def predict(inputVal, input_nodes, hidden_nodes, output_nodes):
    for i in range(len(input_nodes)):
        if len(inputVal)> i:
            input_nodes[i].activation = inputVal[i]
        else:
            input_nodes[i].activation = 0
    for hidden in hidden_nodes:
        hidden.activation = 0
        hidden.activate(tan)
    for output in output_nodes:
        output.activation = 0
        output.activate(tan)
    return output_nodes

def plot_confusion(matrix):
    xlst = []
    ylst= []
    colors = []
    labels = list(string.ascii_lowercase)
    for x in range(26):
        for y in range(26):
            if matrix[x][y] != 0:
                colors.append(matrix[x][y])
                ylst.append(y)
                xlst.append(x)
    plt.scatter(xlst, ylst, c=colors, cmap=cmaps.viridis_r)
    plt.ylabel('Predicted Letter')
    plt.xlabel('Correct Letter')
    xticks = numpy.arange(26)
    plt.xticks(xticks, labels)
    plt.yticks(xticks, labels)
    plt.title("Confusion Matrix Graph")
    plt.colorbar()
    plt.show()
