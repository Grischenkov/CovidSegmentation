import tensorflow as tf

def model():
    resNet = tf.keras.applications.ResNet50(include_top=False, input_shape=(512, 512, 1), weights=None)
    model = tf.keras.models.Sequential()
    model.add(resNet)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    return model