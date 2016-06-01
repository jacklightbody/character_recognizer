import character_recognizer as cr
import numpy
inputs, lists, outputs = cr.load_chars("letter.data")
train_inputs = inputs[:int(len(inputs)/4)]
context_train_inputs = cr.compute_next_prev(lists[:int(len(inputs)/4)], outputs[:int(len(inputs)/4)])
train_outputs = outputs[:int(len(outputs)/4)]
context_test_inputs = cr.compute_next_prev(lists[int(len(inputs)/4):int(2*len(inputs)/4)], outputs[int(len(inputs)/4):int(2*len(inputs)/4)])
test_inputs = inputs[int(len(inputs)/4):int(2*len(inputs)/4)]
test_outputs = outputs[int(len(inputs)/4):int(2*len(inputs)/4)]
second_test_inputs = inputs[int(3*len(inputs)/4):int(4*len(inputs)/4)]
second_context_test_inputs = cr.compute_next_prev(lists[int(3*len(inputs)/4):int(4*len(inputs)/4)], outputs[int(3*len(inputs)/4):int(4*len(inputs)/4)])
second_test_outputs = outputs[int(3*len(inputs)/4):int(4*len(inputs)/4)]
weights0, weights1 = cr.learn2Init(context_train_inputs, train_outputs, 50, 20, .075)

print "Learned Pixels"

predict_outputs_learning = cr.predictOutputs(context_test_inputs,  weights0, weights1, cr.multinomial_output)

confusion_matrix = cr.get_confusion_matrix(predict_outputs_learning, test_outputs)
print confusion_matrix
cr.plot_confusion(confusion_matrix)
'''
second_train_inputs = cr.compute_context(lists[int(len(inputs)/4):int(2*len(inputs)/4)], predict_outputs_learning)

weights0, weights1 = cr.learn2Init(context_train_inputs, outputs[:int(len(inputs)/10)], 10, 20, .075)

weights0, weights1 = cr.learn2(second_train_inputs.T, numpy.asarray(test_outputs).T, 50, 20, .075, weights0, weights1)



print "Learned Context"
predict_outputs_learning = cr.predictOutputs(second_context_test_inputs, weights0, weights1, cr.multinomial_output)
print "iter=50, hid=20, eta=.075"
confusion_matrix = cr.get_confusion_matrix(predict_outputs_learning, second_test_outputs)
#print confusion_matrix

weights0, weights1 = cr.learn2(second_train_inputs.T, numpy.asarray(test_outputs).T, 50, 25, .075, weights0, weights1)
predict_outputs_learning = cr.predictOutputs(second_context_test_inputs, weights0, weights1, cr.multinomial_output)
print "iter=50, hid=25, eta=.075"
confusion_matrix = cr.get_confusion_matrix(predict_outputs_learning, second_test_outputs)
#print confusion_matrix
cr.plot_confusion(confusion_matrix)

weights0, weights1 = cr.learn2(second_train_inputs.T, numpy.asarray(test_outputs).T, 50, 20, .075, weights0, weights1)
predict_outputs_learning = cr.predictOutputs(second_context_test_inputs, weights0, weights1, cr.multinomial_output)
print "iter=50, hid=20, eta=.05"
confusion_matrix = cr.get_confusion_matrix(predict_outputs_learning, second_test_outputs)
#print confusion_matrix
cr.plot_confusion(confusion_matrix)'''
