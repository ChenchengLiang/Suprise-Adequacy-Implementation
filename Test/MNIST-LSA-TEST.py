'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from scipy import stats
import numpy as np
from sklearn.neighbors.kde import KernelDensity
number_samples=1001
batch_size = int(number_samples/5)
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#normalization
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

## trim data size
x_train=x_train[0:number_samples]
y_train=y_train[0:number_samples]
x_test=x_test[0:number_samples]
y_test=y_test[0:number_samples]
print('--------------------')
print('trimmed x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
##

##  model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

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
#print('Test mode layer', layer_number, 'value:\n',layer_output)


#print('DEBUG:',layer_output_from_test.size/layer_output_from_test.shape[0])
#neuron_number=layer_output_from_test.size/layer_output_from_test.shape[0]
#Test=np.reshape(layer_output_from_test,(layer_output_from_test.shape[0],int(neuron_number)))
#print('DEBUG:',Test.shape)
#layer_output_from_test=layer_output_from_test[1:].flatten()


# output in train mode = 1
layer_output_from_train = get_layer_output([x_train, 1])[0]
print('Train mode layer', layer_number, 'output shape',layer_output_from_train.shape)
#print('Test mode layer', layer_number, 'value:\n',layer_output)


##reshape neuron array before KDE
neuron_number=layer_output_from_test.size/layer_output_from_test.shape[0]
layer_output_from_test=np.reshape(layer_output_from_test,(layer_output_from_test.shape[0],int(neuron_number)))
print('Re-shaped layer_output_from_test', layer_output_from_test.shape)
neuron_number=layer_output_from_train.size/layer_output_from_train.shape[0]
layer_output_from_train=np.reshape(layer_output_from_train,(layer_output_from_train.shape[0],int(neuron_number)))
print('Re-shaped layer_output_from_train', layer_output_from_train.shape)



## KDE
#reference https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity.score
#reference https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#scipy.stats.gaussian_kde
scotts_factor=layer_output_from_train[0:number_samples-1].size**(-1./(layer_output_from_train.ndim+4))
print('Bandwidth scotts_factor:',scotts_factor)
kde = KernelDensity(kernel='gaussian', bandwidth=scotts_factor).fit(layer_output_from_train[0:number_samples-1])
kde_score=kde.score(layer_output_from_train[number_samples-1:])/neuron_number
print('KDE score:', kde_score)
#kde_sample=kde.score_samples(layer_output_from_train)
#print('KDE score samples:', kde_sample)
#kde.score_samples(layer_output_from_train)

#kernel = stats.gaussian_kde(layer_output_from_train[0:number_samples-1])
#print('kernel:',kernel.evaluate(layer_output_from_train[0:number_samples-1]))

## LSA
LSA=-kde_score
#LSA=-kde_sample
print('LSA_sklearn:', LSA)
#print('LSA_scipy:', stats.gaussian_kde(layer_output_from_train))

##

