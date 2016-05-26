import character_recognizer as cr
import numpy
inputs, outputs = cr.load_chars("letter.data")
train_inputs = inputs[:int(len(inputs)/20)]
train_outputs = outputs[:int(len(outputs)/20)]
test_inputs = inputs[int(160*len(inputs)/200):]
test_outputs = outputs[int(160*len(outputs)/200):]
input_layer, hidden_layer, output_layer = cr.learn(train_inputs, train_outputs, 20, 2, .0001)
print "Learned"
predict_outputs = cr.predictOutputs(test_inputs, input_layer, hidden_layer, output_layer, cr.multinomial_output)
confusion_matrix = cr.get_confusion_matrix(predict_outputs, test_outputs)
print(confusion_matrix)