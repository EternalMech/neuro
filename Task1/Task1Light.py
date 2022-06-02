from tensorflow.keras.datasets import mnist     #Библиотека с базой Mnist
from tensorflow.keras.models import Sequential  # Подлючаем класс создания модели Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout       # Подключаем класс Dense - полносвязный слой
from tensorflow.keras.optimizers import Adam    # Подключаем оптимизатор Adam
from tensorflow.keras import utils              #Утилиты для to_categorical
import numpy as np                              # Подключаем библиотеку numpy
from tensorflow.keras.preprocessing import image #Для отрисовки изображения
import matplotlib.pyplot as plt # Отрисовка графика
from PIL import Image

# Загрузка данных Mnist
(x_train_org, y_train_org), (x_test, y_test) = mnist.load_data()


# Обучающая выборка 50.000 примеров
x_train = x_train_org[:50000].reshape(50000, 784).astype('float32') / 255
y_train = utils.to_categorical(y_train_org[:50000], 10)

x_test = x_test.reshape(10000, 784).astype('float32') / 255
y_test = utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(400, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history0 = model.fit(x_train,
                    y_train,
                    batch_size=64,
                    epochs=20,
                    verbose=1,
                    validation_data = (x_test, y_test))


# Обучающая выборка 10.000 примеров
x_train = x_train_org[:10000].reshape(10000, 784).astype('float32') / 255
y_train = utils.to_categorical(y_train_org[:10000], 10)

model = Sequential()
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(400, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history1 = model.fit(x_train,
                    y_train,
                    batch_size=64,
                    epochs=20,
                    verbose=1,
                    validation_data = (x_test, y_test))


# Обучающая выборка 500 примеров
x_train = x_train_org[:500].reshape(500, 784).astype('float32') / 255
y_train = utils.to_categorical(y_train_org[:500], 10)

model = Sequential()
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(400, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history2 = model.fit(x_train,
                    y_train,
                    batch_size=64,
                    epochs=20,
                    verbose=1,
                    validation_data = (x_test, y_test))


#Запишем точности нейронных сетей с различными размерами обучающей выборки
print('Обучающая выборка 50000 примеров: ', round(history0.history['val_accuracy'][-1]*100, 2), '%', sep = '')

print('Обучающая выборка 10000 примеров: ', round(history1.history['val_accuracy'][-1]*100, 2), '%', sep = '')

print('Обучающая выборка 500 примеров: ', round(history2.history['val_accuracy'][-1]*100, 2), '%', sep = '')


#Выводим графики обучения

print('Обучающая выборка 50000 примеров: ')

plt.plot(history0.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history0.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

plt.plot(history0.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history0.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()


print('Обучающая выборка 10000 примеров: ')

plt.plot(history1.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history1.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

plt.plot(history1.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history1.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()


print('Обучающая выборка 500 примеров: ')

plt.plot(history2.history['loss'],
         label='Ошибка на обучающем наборе')
plt.plot(history2.history['val_loss'],
         label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.grid()
plt.show()

plt.plot(history2.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history2.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.grid()
plt.show()
