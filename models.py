import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, concatenate,
    Input, Reshape, BatchNormalization,
    GlobalAveragePooling2D, Dropout, Dense
)


def CNN_model(input_shape=(99, 13)):

    input_layer = keras.Input(shape=(99, 13))  

    # Add new axis to input layer
    reshape_layer = Reshape(input_shape=input_shape, target_shape=(99, 13, 1))(input_layer)

    # 1st conv layer
    x = Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(reshape_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same', strides=(2, 2))(x)

    # 2nd conv layer
    x = Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3,3), padding='same', strides=(2, 2))(x)

    # 3rd conv layer
    x = Conv2D(32, (2,2), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same', strides=(2, 2))(x)

    # flatten output and feed into dense layer
    x = tf.keras.layers.Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    # softmax output layer
    x = Dense(20, activation='softmax')(x)

    model = Model(input_layer, x)
    return model


def inceptionModule(x_input, n):

    # Convolution layer 1x1
    conv_1x1 = Conv2D(n, (1,1), padding='same', activation='relu')(x_input)
    
    # Convolution layer 3x3
    conv_3x3 = Conv2D(n, (1,1), padding='same', activation='relu')(x_input)
    conv_3x3 = Conv2D(n, (1,1), padding='same', activation='relu')(conv_3x3)
    
    # Convolution layer 5x5
    conv_5x5 = Conv2D(n, (1,1), padding='same', activation='relu')(x_input)
    conv_5x5 = Conv2D(n, (1,1), padding='same', activation='relu')(conv_5x5)
    
    # pool + proj
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x_input)
    pool = Conv2D(n, (1,1), padding='same', activation='relu')(pool)
    
    # Output layer
    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool], axis=3)
    
    return output


def Inception(input_shape=(99,13)):
    
    input_layer = Input(shape=(99, 13))
    
    reshape_layer = Reshape(input_shape=input_shape, target_shape=(99, 13, 1))(input_layer)
      
    x = Conv2D(32, (3,3), padding='same', strides=(2, 2), activation='relu')(reshape_layer)
    x = MaxPooling2D((2,2), padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    
    x = inceptionModule(x, 32)
    x = inceptionModule(x, 64)
    x = MaxPooling2D((2,2))(x)
    
    x = inceptionModule(x, 64)
    x = inceptionModule(x, 128)
    
    x = GlobalAveragePooling2D()(x)   
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu')(x)
    
    x = Dense(35, activation='softmax')(x)
    
    model = Model(input_layer, x)
    return model


def attRNN(input_shape=(99, 13),
                  rnn='LSTM',
                  multi_rnn=True,
                  attention=True,
                  dropout=0.2):


    # Fetch input
    inputs = tf.keras.Input(shape=input_shape)
    reshape = tf.keras.layers.Reshape(
        input_shape=input_shape, target_shape=(99, 13, 1))(inputs)

    # Normalization Layer
    layer_out = tf.keras.layers.BatchNormalization()(reshape)

    # Convolutional Layer
    # 35: number of categories 
    layer_out = tf.keras.layers.Conv2D(35, kernel_size=(10, 4),
                                       padding='same', activation='relu')(layer_out)
    layer_out = tf.keras.layers.BatchNormalization()(layer_out)
    layer_out = tf.keras.layers.Conv2D(1, kernel_size=(6, 5),
                                       padding='same', activation='relu')(layer_out)
    layer_out = tf.keras.layers.BatchNormalization()(layer_out)
    layer_out = tf.keras.layers.Lambda(
        lambda x: tf.keras.backend.squeeze(x, -1), name='squeeze_dim')(layer_out)

    # LSTM Layer
    if rnn == 'LSTM':
        layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            40, return_sequences=True, dropout=dropout))(layer_out)
        if multi_rnn:
            layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                40, return_sequences=True, dropout=dropout))(layer_out)

    # GRU Layer
    if rnn == 'GRU':
        layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            40, return_sequences=True, dropout=dropout))(layer_out)
        if multi_rnn:
            layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                40, return_sequences=True, dropout=dropout))(layer_out)

    # Attention Layer
    if attention:
        query, value = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=2, axis=2))(layer_out)
        layer_out = tf.keras.layers.Attention(name='Attention')([query, value])

    # Classification Layer
    outputs = tf.keras.layers.Flatten()(layer_out)
    outputs = tf.keras.layers.Dense(64, activation='relu')(outputs)
    outputs = tf.keras.layers.Dense(35, activation='softmax')(outputs)

    # Output Layer
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model


def DSCNN(input_shape=(99, 13)):
    inputs = keras.Input(shape=(99, 13))

    # Add new dimension to input layer
    reshape_layer = layers.Reshape(input_shape=input_shape, target_shape=(99, 13, 1))(inputs)

    # 2D Convolutional layer
    x = layers.Conv2D(filters=128,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      activation='relu',
                      padding='same',
                      kernel_initializer="he_normal")(reshape_layer)
    x = layers.BatchNormalization()(x)

    # Depthwise Layer 1
    x = layers.DepthwiseConv2D(kernel_size=(2, 2),
                               strides=(2, 2),
                               activation=None,
                               padding="valid",
                               depthwise_initializer="glorot_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.Conv2D(filters=128,
                      kernel_size=1,
                      strides=(1, 1),
                      activation=None,
                      padding='valid',
                      kernel_initializer="glorot_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation='relu')(x)

    # Depthwise Layer 2
    x = layers.DepthwiseConv2D(kernel_size=(2, 2),
                               strides=(1, 1),
                               activation=None,
                               padding="valid",
                               depthwise_initializer="glorot_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.Conv2D(filters=128,
                      kernel_size=1,
                      strides=(1, 1),
                      activation=None,
                      padding='valid',
                      kernel_initializer="glorot_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation='relu')(x)

    # Depthwise Layer 3
    x = layers.DepthwiseConv2D(kernel_size=(2, 2),
                               strides=(1, 1),
                               activation=None,
                               padding="valid",
                               depthwise_initializer="glorot_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.Conv2D(filters=276,
                      kernel_size=1,
                      strides=(1, 1),
                      activation=None,
                      padding='valid',
                      kernel_initializer="glorot_uniform")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation='relu')(x)

    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)

    x = layers.Flatten()(x)

    # Classification Layer
    output = layers.Dense(20,
                          activation="softmax",
                          kernel_initializer="glorot_uniform")(x)

    # Output Layer
    model = keras.Model(inputs=inputs, outputs=output)

    return model