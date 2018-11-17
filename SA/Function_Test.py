from SC import suprise_coverage
from Dataset_MNIST import get_mnist_data, num_classes, input_shape
from CNN_MNIST import train_mnist_cnn_model
from AT import activation_trace
from Trim_Data_Size import trim_data
from DSA import distance_based_suprise_adequacy
import numpy as np



'''
##SC.py test (finished)

ub=2 #upper bound
n=10 #number of buckets

DSA=np.random.rand(10)

suprise_coverage(DSA,ub,n)

'''

'''
##Dataset_MNIST.py test (finished)
x_train, y_train, x_test, y_test = get_mnist_data()

'''

'''
##CNN_MNIST test.py (finished)
#input data
x_train, y_train, x_test, y_test = get_mnist_data() 

number_samples=100
start_point=30
x_train,y_train,x_test,y_test = trim_data(start_point,number_samples,x_train,y_train,x_test,y_test)


batch_size = int(number_samples/5)
epochs = 1
model = train_mnist_cnn_model(batch_size, epochs, x_train, y_train, x_test, y_test, num_classes, input_shape)
'''



'''
##LSA.py test (to do)
'''

'''
##AT.py test (finished)
x_train, y_train, x_test, y_test = get_mnist_data() 

number_samples=100
start_point=30
x_train,y_train,x_test,y_test = trim_data(start_point,number_samples,x_train,y_train,x_test,y_test)


batch_size = int(number_samples/5)
epochs = 1
model = train_mnist_cnn_model(batch_size, epochs, x_train, y_train, x_test, y_test, num_classes, input_shape)
AT_train=activation_trace(model,x_train,1)
AT_test=activation_trace(model,x_test,0)
'''

'''
##DSA.py test (fnished)
x_train, y_train, x_test, y_test = get_mnist_data() 

number_samples=100
start_point=30
x_train,y_train,x_test,y_test = trim_data(start_point,number_samples,x_train,y_train,x_test,y_test)


batch_size = int(number_samples/5)
epochs = 1
model = train_mnist_cnn_model(batch_size, epochs, x_train, y_train, x_test, y_test, num_classes, input_shape)
AT_train=activation_trace(model,x_train,1)
AT_test=activation_trace(model,x_test,0)

C=3
DSA=distance_based_suprise_adequacy(AT_train,AT_test,y_test,C)
print('DSA shape', DSA.shape)
print('DSA',DSA)
'''






