import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

if __name__ == '__main__':

    save_path = "/home/jorge/PycharmProjects/AudioExtraction/result_graphs/"
    # labels_path = "C:\\Users\\jorge\\DatasetsTFM\\KaggleData\\train.csv"
    # train_path = "C:\\Users\\jorge\\DatasetsTFM\\KaggleData\\train"
    numpy_save_path = "/home/jorge/PycharmProjects/AudioExtraction/numpy_data/"

    tag = 'prueba_100'
    print('Loading data, started at {}'.format(datetime.datetime.now()))

    X_train = np.load(os.path.join(numpy_save_path, 'xtrain.npy'))
    X_test = np.load(os.path.join(numpy_save_path, 'xtest.npy'))
    Y_train = np.load(os.path.join(numpy_save_path, 'ytrain.npy'))
    Y_test = np.load(os.path.join(numpy_save_path, 'ytest.npy'))

    print('Setting parameters and data reshape for the training')
    # opt = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    opt = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    Y_train, Y_test = Y_train.astype(int), Y_test.astype(int)
    Y_train, Y_test = tf.keras.utils.to_categorical(Y_train, 2), tf.keras.utils.to_categorical(Y_test, 2)

    # Normalization of the data
    # X_norm = np.linalg.norm(X_train)
    # print(X_train.max().max())
    X_norm = X_train.max().max()

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1) / X_norm
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1) / X_norm

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(20, kernel_size=(7, 7), activation=tf.nn.relu, input_shape=X_train.shape[1:]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(40, kernel_size=(7, 7), activation=tf.nn.relu, name='Conv2'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu, name="Dense1"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax, name="Softmax")
        ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print('Training model')
    history = model.fit(X_train, Y_train, epochs=10, verbose=1)
    score = model.evaluate(X_test, Y_test)

    print(score)
    print('Finished at {}, saving the results as graphs.'.format(datetime.datetime.now()))

    #Accuracy plot
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')

    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(save_path + tag + 'model_accuracy' + str(score[1]) + '.pdf')
    plt.close()
    #Loss plot
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(save_path + tag + 'model_loss' + str(score[1]) + '.pdf')

    #
    # from sklearn.metrics import classification_report,confusion_matrix
    # import numpy as np
    # #Compute probabilities
    # y_train = data_generator.classes
    # Y_pred = nn.predict_generator(data_generator, steps=80)
    # # print(Y_pred.shape)
    # #Assign most probable label
    # y_pred = np.argmax(Y_pred, axis=1)
    # #Plot statistics
    # print('Analysis of results')
    # target_names = ['no_whale', 'whale']
    # print(classification_report(y_train, y_pred, target_names=target_names))
    # print(confusion_matrix(y_train, y_pred))

    # #Confusion Matrix
    # from sklearn.metrics import classification_report,confusion_matrix
    # import numpy as np
    # #Compute probabilities
    # y_test = test_gen.classes
    # Y_pred = nn.predict_generator(test_gen, steps=20)
    # # print(Y_pred.shape)
    # #Assign most probable label
    # y_pred = np.argmax(Y_pred, axis=1)
    # #Plot statistics
    # print('Analysis of results')
    # target_names = ['no_whale', 'whale']
    # print(classification_report(y_test, y_pred,target_names=target_names))
    # print(confusion_matrix(y_test, y_pred))
