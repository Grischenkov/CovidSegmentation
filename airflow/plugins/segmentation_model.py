import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Input, Conv2DTranspose, concatenate

def model() :
    x_input = Input((512, 512, 1))

    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)
    x_1 = BatchNormalization()(x)
    x = MaxPooling2D((2, 2)) (x_1) 
    x = Dropout(0.2)(x) 

    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)
    x_2 = BatchNormalization()(x)
    x = MaxPooling2D((2, 2)) (x_2) 
    x = Dropout(0.2)(x) 

    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((1, 1)) (x) 
    x = Dropout(0.2)(x) 

    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((1, 1)) (x) 
    x = Dropout(0.2)(x) 

    x = BatchNormalization() (x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x) 
    
    x = Conv2DTranspose(128, (2, 2), strides=(2,2), padding='same') (x)
    x = BatchNormalization() (x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x) 

    x = Conv2DTranspose(64, (2, 2), padding='same') (x)
    x = concatenate([x, x_2])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)

    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (x)
    x = concatenate([x, x_1], axis=3)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (x)

    infection_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='infect_output') (x)

    model = tf.keras.Model(inputs=x_input, outputs=infection_segmentation, name='segmenter_model')
        
    return model

def dice(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    return (2. * tf.reduce_sum(y_true_f * y_pred_f)) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f))

def dice_loss(y_true, y_pred):
    loss = 1 - dice(y_true, y_pred)
    return loss