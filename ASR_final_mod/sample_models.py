from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, CuDNNGRU, MaxPooling1D, Dropout)

def simple_rnn_model(input_dim, output_dim=29):

    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, output_dim=29):
    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnn = CuDNNGRU(units, return_sequences=True, name='rnn')(input_data)
    bn_rnn = BatchNormalization()(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    simp_rnn = CuDNNGRU(units, return_sequences=True, name='rnn')(bn_cnn)
    bn_rnn = BatchNormalization()(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
   
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
 
    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnn = CuDNNGRU(units, return_sequences=True)(input_data)
    bn_rnn = BatchNormalization()(simp_rnn)
    if recur_layers > 1:
        for i in range(recur_layers -1):
            simp_rnn = CuDNNGRU(units, return_sequences=True)(bn_rnn)
            bn_rnn = BatchNormalization()(simp_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, recur_layers, output_dim=29):
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    mp_cnn = MaxPooling1D()(bn_cnn)
    do_cnn = Dropout(0.2)(mp_cnn)
    bidir_rnn = Bidirectional(CuDNNGRU(units, return_sequences=True))(do_cnn)
    bn_rnn = BatchNormalization()(bidir_rnn)
    do_rnn = Dropout(0.2)(bn_rnn)
    if recur_layers > 1:
        for i in range(recur_layers -1):
            bidir_rnn = Bidirectional(CuDNNGRU(units, return_sequences=True))(do_rnn)
            bn_rnn = BatchNormalization()(bidir_rnn)
            do_rnn = Dropout(0.2)(bn_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(do_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length =  lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)//2
    print(model.summary())
    return model