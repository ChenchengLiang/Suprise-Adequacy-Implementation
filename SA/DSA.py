import keras
from keras import backend as K
from Dataset_MNIST import num_classes
import numpy as np
def distance_based_suprise_adequacy(AT_train,AT_test,y_test,C):
    
    
    print('AT_train shape', AT_train.shape)
    print('AT_test shape', AT_test.shape)
    C=keras.utils.to_categorical(C, num_classes)
    print('Class',np.argmax(C),'value', C)
    DSA=np.array([[0,0]])
    
    #index k for every new input
    
    for k in range(AT_test.shape[0]):
        ##argmin AT(x)-AT(x_i) compute dist_a
        t_a=np.array([])
        for i in range(AT_train.shape[0]):
            if(np.array_equal(y_test[i],C)):
                t_a=np.append(t_a,np.linalg.norm(AT_train[i]-AT_test[k]))
            else:
                t_a=np.append(t_a,9999)
                
        #print('t_a shape',t_a.shape)
        
        min_index=np.argmin(t_a)
        #print('min index',min_index,'value',t_a[min_index])
        #print('AT_a',AT_train[min_index])
        
        dist_a=np.linalg.norm(AT_test[k]-AT_train[min_index])
        #print('dist_a',dist_a)
        
        
        ###argmin AT(x)-AT(x_i) compute dist_b
        t_b=np.array([])
        for i in range(AT_train.shape[0]):
            if(np.array_equal(y_test[i],C)):
                t_b=np.append(t_b,9999)
            else:
                t_b=np.append(t_b,np.linalg.norm(AT_train[i]-AT_test[k]))
                
                
        #print('t_b shape',t_a.shape)
        
        min_index=np.argmin(t_b)
        #print('min index',min_index,'value',t_b[min_index])
        #print('AT_a',AT_train[min_index])
        
        dist_b=np.linalg.norm(AT_test[k]-AT_train[min_index])
        #print('dist_b',dist_b)
        
        
        ##DSA = dist_a/dist_b
        
        c_temp=np.argmax(y_test[k])
        #print('c_temp',c_temp)
        temp=np.array([[dist_b/dist_a,c_temp]])
        DSA=np.concatenate((DSA,temp),axis=0)
        
        #print('DSA',DSA)
        #print('Class',c_temp,'category',C)
    
    DSA=DSA[1:]
    return DSA