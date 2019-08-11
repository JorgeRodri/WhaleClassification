import tensorflow as tf


def get_nn_redux():
    model = tf.keras.Sequential()
    return model


def get_nn(x_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(20, kernel_size=(7, 7), activation=tf.nn.relu, input_shape=x_shape))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(40, kernel_size=(7, 7), activation=tf.nn.relu, name='Conv2'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, name="Dense1"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="Softmax"))

    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    return model


def winners_nn(input_size):
    nn = tf.keras.Sequential()
    nn.add(tf.keras.layers.Conv2D(20, kernel_size=(7, 7), activation='relu', input_shape=input_size))
    nn.add(tf.keras.layers.Dropout(0.2))
    nn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    nn.add(tf.keras.layers.Conv2D(40, kernel_size=(7, 7), activation='relu'))
    nn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    nn.add(tf.keras.layers.Flatten())
    nn.add(tf.keras.layers.Dense(512, activation='relu'))
    nn.add(tf.keras.layers.Dropout(0.6))
    nn.add(tf.keras.layers.Dense(2, activation='softmax'))
    return nn


