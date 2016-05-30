import character_recognizer as cr
import numpy
inputs, lists, outputs = cr.load_chars("letter.data")
train_inputs = inputs[:int(len(inputs)/50)]
#context_train_inputs = cr.compute_next_prev(lists[:int(len(inputs)/2)], outputs[:int(len(inputs)/2)])
train_outputs = outputs[:int(len(outputs)/50)]
test_inputs = inputs[int(2*len(inputs)/50):int(3*len(inputs)/50)]
#context_test_inputs = cr.compute_next_prev(lists[int(45*len(inputs)/50):], outputs[int(45*len(inputs)/50):])
test_outputs = outputs[int(2*len(inputs)/50):int(3*len(inputs)/50)]
second_test_inputs = inputs[int(3*len(inputs)/50):int(4*len(inputs)/50)]
second_test_outputs = outputs[int(3*len(inputs)/50):int(4*len(inputs)/50)]
input_layer, hidden_layer, output_layer = cr.learn(train_inputs, train_outputs, 5, 3, .0001)

print "Learned Pixels"

predict_outputs_learning = cr.predictOutputs(test_inputs, input_layer, hidden_layer, output_layer, cr.multinomial_output)
second_train_inputs = cr.compute_context(lists[int(2*len(inputs)/50):int(3*len(inputs)/50)], outputs[int(2*len(inputs)/50):int(3*len(inputs)/50)])


input_layer, hidden_layer, output_layer = cr.learn(second_train_inputs, test_outputs, 5, 3, .0001)



print "Learned"
predict_outputs_learning = cr.predictOutputs(second_test_inputs, input_layer, hidden_layer, output_layer, cr.multinomial_output)
confusion_matrix = cr.get_confusion_matrix(predict_outputs_learning, second_test_outputs)
print(confusion_matrix)
