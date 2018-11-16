'''Trains a simple convnet on the MNIST dataset.
Details of MNIST processing with Keras
https://www.kaggle.com/adityaecdrid/mnist-with-keras-for-beginners-99457#
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
number_samples=101
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
start_point=10
x_train=x_train[start_point:start_point+number_samples]
y_train=y_train[start_point:start_point+number_samples]
x_test=x_test[start_point:start_point+number_samples]
y_test=y_test[start_point:start_point+number_samples]
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
'''
##Evaluation

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
## compute AT from train data
total_layer_number=len(model.layers[:])
print('Totoal Layer:',total_layer_number)

get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[1].output])
AT=get_layer_output([x_train, 1])[0]
print('Train mode layer 1 output shape',AT.shape)                                  

for i in range(2,total_layer_number):
	get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[i].output])
	layer_output_from_train = get_layer_output([x_train, 1])[0]
	print('Train mode layer', i, 'output shape',layer_output_from_train.shape)
		
	neuron_number=layer_output_from_train.size/layer_output_from_train.shape[0]
	layer_output_from_train=np.reshape(layer_output_from_train,(layer_output_from_train.shape[0],int(neuron_number)))
	#print('re-shaped layer_output_from_train:',layer_output_from_train.shape)
	
	neuron_number=AT.size/AT.shape[0]
	AT=np.reshape(AT,(AT.shape[0],int(neuron_number)))
	#print('re-shaped AT:',AT.shape)
	
	
	#print('before concatenate AT shape:',AT.shape)
	AT=np.concatenate((AT,layer_output_from_train), axis=1)
	#print('after concatenate AT shape:',AT.shape)                              

print('AT_train shape', AT.shape)


## compute AT from test data
total_layer_number=len(model.layers[:])
print('Totoal Layer:',total_layer_number)

get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[1].output])
AT_test=get_layer_output([x_test, 0])[0]
print('Test mode layer 1 output shape',AT_test.shape)                                  

for i in range(2,total_layer_number):
	get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[i].output])
	layer_output_from_test = get_layer_output([x_test, 0])[0]
	print('Test mode layer', i, 'output shape',layer_output_from_test.shape)
		
	neuron_number=layer_output_from_test.size/layer_output_from_test.shape[0]
	layer_output_from_test=np.reshape(layer_output_from_test,(layer_output_from_test.shape[0],int(neuron_number)))
	#print('re-shaped layer_output_from_test:',layer_output_from_test.shape)
	
	neuron_number=AT_test.size/AT_test.shape[0]
	AT_test=np.reshape(AT_test,(AT_test.shape[0],int(neuron_number)))
	#print('re-shaped AT_test:',AT_test.shape)
	
	
	#print('before concatenate AT shape:',AT_test.shape)
	AT_test=np.concatenate((AT_test,layer_output_from_test), axis=1)
	#print('after concatenate AT_test shape:',AT_test.shape)                              

print('AT_test shape', AT_test.shape)



##argmin AT(x)-AT(x_i) compute dist_a

C=2
C=keras.utils.to_categorical(C, num_classes)
print('Class', C)
#print('y_train[1]',y_train[1])

t_a=np.array([])
for i in range(AT.shape[0]-1):
	if(np.array_equal(y_train[i],C)):
		t_a=np.append(t_a,np.linalg.norm(AT[i]-AT[-1]))
	else:
		t_a=np.append(t_a,9999)
		
print('t_a shape',t_a.shape)

min_index=np.argmin(t_a)
print('min index',min_index,'value',t_a[min_index])
print('AT_a',AT[min_index])

dist_a=np.linalg.norm(AT[-1]-AT[min_index])
print('dist_a',dist_a)

###argmin AT(x)-AT(x_i) compute dist_b
t_b=np.array([])
for i in range(AT.shape[0]-1):
	if(np.array_equal(y_train[i],C)):
		t_b=np.append(t_b,9999)
	else:
		t_b=np.append(t_b,np.linalg.norm(AT[i]-AT[-1]))
		
		
print('t_b shape',t_a.shape)

min_index=np.argmin(t_b)
print('min index',min_index,'value',t_b[min_index])
print('AT_a',AT[min_index])

dist_b=np.linalg.norm(AT[-1]-AT[min_index])
print('dist_b',dist_b)


##DSA = dist_a/dist_b
DSA=dist_a/dist_b
print('DSA',DSA)
print('Class',C)
