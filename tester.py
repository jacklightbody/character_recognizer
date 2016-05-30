import character_recognizer as cr
import numpy
inputs, lists, outputs = cr.load_chars("letter.data")
train_inputs = inputs[:int(len(inputs)/10)]
#context_train_inputs = cr.compute_context(lists[:int(len(inputs)/50)], outputs[:int(len(inputs)/50)])
train_outputs = outputs[:int(len(outputs)/10)]
test_inputs = inputs[int(2*len(inputs)/10):int(3*len(inputs)/10)]
test_outputs = outputs[int(2*len(inputs)/10):int(3*len(inputs)/10)]
second_test_inputs = inputs[int(3*len(inputs)/10):int(4*len(inputs)/10)]
second_test_outputs = outputs[int(3*len(inputs)/10):int(4*len(inputs)/10)]
input_layer, hidden_layer, output_layer = cr.learnInit(train_inputs, train_outputs, 50, 3, .0001)

print "Learned Pixels"
'''
predict_outputs_learning = cr.predictOutputs(test_inputs, input_layer, hidden_layer, output_layer, cr.multinomial_output)
second_train_inputs = cr.compute_context(lists[int(2*len(inputs)/20):int(3*len(inputs)/20)], predict_outputs_learning)

context_input_layer, context_hidden_layer, context_output_layer = cr.learnInit(context_train_inputs, outputs[:int(len(inputs)/50)], 5, 3, .0001)

input_layer, hidden_layer, output_layer = cr.learn(context_input_layer, context_hidden_layer, context_output_layer, second_train_inputs, test_outputs, 20, 3, .0001)
'''


print "Learned Context"
predict_outputs_learning = cr.predictOutputs(second_test_inputs, input_layer, hidden_layer, output_layer, cr.multinomial_output)
confusion_matrix = cr.get_confusion_matrix(predict_outputs_learning, second_test_outputs)
print(confusion_matrix)
