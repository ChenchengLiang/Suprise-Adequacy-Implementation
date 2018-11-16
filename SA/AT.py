from keras import backend as K
import numpy as np
# flag =0 for test 
# flag =1 for training
def activation_trace(model,data,flag):
    if(flag==0):
        str='test'
    else:
        str='train'
    ## compute AT from train data
    total_layer_number=len(model.layers[:])
    print('Totoal Layer:',total_layer_number)
    
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[1].output])
    AT=get_layer_output([data, flag])[0]
    print('Layer 1 output shape',AT.shape)                                  
    
    for i in range(2,total_layer_number):
        get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[i].output])
        layer_output_from_train = get_layer_output([data, flag])[0]
        print('Layer', i, 'output shape',layer_output_from_train.shape)
            
        neuron_number=layer_output_from_train.size/layer_output_from_train.shape[0]
        layer_output_from_train=np.reshape(layer_output_from_train,(layer_output_from_train.shape[0],int(neuron_number)))
        #print('re-shaped layer_output_from_train:',layer_output_from_train.shape)
        
        neuron_number=AT.size/AT.shape[0]
        AT=np.reshape(AT,(AT.shape[0],int(neuron_number)))
        #print('re-shaped AT:',AT.shape)
        
        
        #print('before concatenate AT shape:',AT.shape)
        AT=np.concatenate((AT,layer_output_from_train), axis=1)
        #print('after concatenate AT shape:',AT.shape)                              
    
    print(str,'AT shape', AT.shape)
    return AT      