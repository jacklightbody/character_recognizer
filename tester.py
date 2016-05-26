import character_recognizer as cr
import numpy
pixels, nexts, outputs = cr.load_chars("letter.data")
train_inputs = inputs[:int(len(inputs)/30)]
context_train_inputs = nexts[:int(len(nexts)/30)]
train_outputs = outputs[:int(len(outputs)/30)]
test_inputs = inputs[int(15*len(inputs)/20):]
test_outputs = outputs[int(15*len(outputs)/20):]
input_layer, hidden_layer, output_layer = cr.learn(train_inputs, train_outputs, 30, 2, .0001)
print "Learned"
predict_outputs = cr.predictOutputs(test_inputs, input_layer, hidden_layer, output_layer, cr.multinomial_output)
confusion_matrix = cr.get_confusion_matrix(predict_outputs, test_outputs)
print(confusion_matrix)