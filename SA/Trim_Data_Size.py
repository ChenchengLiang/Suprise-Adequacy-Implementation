

def trim_data(start_point,number_samples,x_train,y_train,x_test,y_test):
    ## trim data size

    x_train=x_train[start_point:start_point+number_samples]
    y_train=y_train[start_point:start_point+number_samples]
    x_test=x_test[start_point:start_point+number_samples]
    y_test=y_test[start_point:start_point+number_samples]
    print('--------------------')
    print('start point',start_point)
    print('trimmed x_train shape:', x_train.shape)
    print('trimmed x_test shape:', x_test.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return x_train,y_train,x_test,y_test