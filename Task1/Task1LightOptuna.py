from tensorflow.keras.datasets import mnist     #Библиотека с базой Mnist
from tensorflow.keras.models import Sequential  # Подлючаем класс создания модели Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Conv2D       # Подключаем класс Dense - полносвязный слой
from tensorflow.keras.optimizers import Adam    # Подключаем оптимизатор Adam
from tensorflow.keras import utils              #Утилиты для to_categorical
import matplotlib.pyplot as plt  # Отрисовка графика
import optuna
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import random
import os


seed = 256
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def objective(trial):

    epochs = trial.suggest_int('epoches', 1, 35)
    activation = trial.suggest_categorical('activation', ['relu', 'linear', 'sigmoid', 'softmax'])
    batch_size = trial.suggest_int('batch_size', 16, 60000)
    n_neuro0 = trial.suggest_int('1LayerNeurons', 1, 4000)
    n_neuro1 = trial.suggest_int('2LayerNeurons', 1, 4000)

    (x_train_org, y_train_org), (x_test, y_test) = mnist.load_data()

    x_train = x_train_org.reshape(60000, 784).astype('float32') / 255
    y_train = utils.to_categorical(y_train_org, 10)

    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    y_test = utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Dense(n_neuro0, input_dim=784, activation=activation))
    model.add(Dense(n_neuro1, activation=activation))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.3)

    results = model.evaluate(x_test, y_test, batch_size=128)

    return results[1]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)


