import keras
from keras import backend as K
import numpy as np
from sklearn.neighbors.kde import KernelDensity


from Dataset_MNIST import get_mnist_data, num_classes, input_shape
from CNN_MNIST import train_mnist_cnn_model



x_train, y_train, x_test, y_test = get_mnist_data()

batch_size = 50 
epochs = 1
## trim data size
x_train=x_train[0:101]
y_train=y_train[0:101]
x_test=x_test[0:101]
y_test=y_test[0:101]
print('--------------------')
print('trimmed x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model = train_mnist_cnn_model(batch_size, epochs, x_train, y_train, x_test, y_test, num_classes, input_shape)

##Evaluation

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




## extract layer output
# reference: https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
layer_number=1;
layer = model.layers[layer_number]
print('Layer information:',layer.output)

get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[layer_number].output])

# output in test mode = 0
layer_output_from_test = get_layer_output([x_test, 0])[0]
print('Test mode layer', layer_number, 'output shape',layer_output_from_test.shape)

# output in train mode = 1
layer_output_from_train = get_layer_output([x_train, 1])[0]
print('Train mode layer', layer_number, 'output shape',layer_output_from_train.shape)

##reshape neuron array before KDE
neuron_number=layer_output_from_test.size/layer_output_from_test.shape[0]
layer_output_from_test=np.reshape(layer_output_from_test,(layer_output_from_test.shape[0],int(neuron_number)))
print('Re-shaped layer_output_from_test', layer_output_from_test.shape)
neuron_number=layer_output_from_train.size/layer_output_from_train.shape[0]
layer_output_from_train=np.reshape(layer_output_from_train,(layer_output_from_train.shape[0],int(neuron_number)))
print('Re-shaped layer_output_from_train', layer_output_from_train.shape)